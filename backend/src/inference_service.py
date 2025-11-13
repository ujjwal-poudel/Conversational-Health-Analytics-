#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference Script for Depression Score Prediction

This script loads a pre-trained multi-target (8-symptom) regression model 
and uses it to predict a single, total depression score from a list of 
transcript turns.

This is intended for a backend service.
"""

import torch
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Tuple

# --- CRITICAL DEPENDENCY ---
# The following import MUST work. This means the file
# 'hdsc/model.py' (or wherever your model class is defined)
# must be available to this script (e.g., in the same directory
# or in your project's Python path).

# Add the src directory to Python path for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

try:
    from hdsc.model import PHQTotalMulticlassAttentionModelBERT
except ImportError as e:
    print("=" * 60)
    print(" IMPORT ERROR: Could not import 'PHQTotalMulticlassAttentionModelBERT' from 'hdsc.model'")
    print("=" * 60)
    print(" Issue Details:")
    print(f"   Import Error: {e}")
    print(f"   Current Python Path: {sys.path}")
    print(f"   Looking for: {os.path.join(src_dir, 'hdsc/model.py')}")
    print(f"   File exists: {os.path.exists(os.path.join(src_dir, 'hdsc/model.py'))}")
    print("")
    print(" For Mac M3 System Setup:")
    print("   1. Install dependencies: pip install torch transformers accelerate")
    print("   2. Ensure virtual environment is activated")
    print("   3. Place model file at: model_2_15.pt")
    print("   4. The src/hdsc/ directory should contain model.py with the class")
    print("")
    print("  This error will prevent the depression scoring API from working.")
    print("=" * 60)
    exit(1)


# --- CONFIGURATION ---
# Set these variables to match your environment.

# Path to the single, best model checkpoint you want to use in production.
MODEL_PATH = "/Volumes/MACBACKUP/models/saved_models/robert_multilabel_no-regression_/model_2_15.pt" 

# Pre-trained tokenizer name (must match what was used in training).
TOKENIZER_NAME = "sentence-transformers/all-distilroberta-v1"


def set_device() -> torch.device:
    """
    Checks for available hardware (MPS for Apple, CUDA for NVIDIA) 
    and returns the appropriate torch.device, defaulting to CPU.
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device


def load_artifacts(
    model_path: str, 
    tokenizer_name: str, 
    device: torch.device
) -> Tuple[PHQTotalMulticlassAttentionModelBERT, AutoTokenizer]:
    """
    Loads the trained model checkpoint and tokenizer from disk.
    This function should be run ONCE when your backend service starts.

    Args:
        model_path: Filepath to the .pt model checkpoint.
        tokenizer_name: Name of the tokenizer from Hugging Face.
        device: The torch.device to load the model onto.

    Returns:
        A tuple containing the loaded (model, tokenizer).
    """
    
    print(f"Loading tokenizer: {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    print(f"Loading model from: {model_path}...")
    # We use map_location=device to ensure the model loads correctly
    # on any hardware, even if it was trained on a different machine (e.g., CUDA).
    try:
        loaded_dict = torch.load(model_path, map_location=device, weights_only=True)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_path}'.")
        print("Please update the 'MODEL_PATH' variable in this script.")
        raise
    
    # Re-create the model structure using saved hyperparameters
    model_kwargs = loaded_dict["kwargs"]
    model_state_dict = loaded_dict["model"]
    
    model = PHQTotalMulticlassAttentionModelBERT(
        device=device,
        **model_kwargs,
    )
    
    # Load the trained weights into the model
    model.load_state_dict(model_state_dict)
    
    # Move model to the correct device (e.g., MPS, CUDA, or CPU)
    model = model.to(device)
    
    # --- CRUCIAL ---
    # Set the model to evaluation mode. This disables
    # operations like dropout, which are only used during training.
    model.eval()
    
    print("Artifacts loaded successfully.")
    return model, tokenizer


@torch.no_grad()  # Decorator to disable gradient calculations
@torch.no_grad()  # Decorator to disable gradient calculations
def get_depression_score(
    transcript_turns: List[str], 
    model: PHQTotalMulticlassAttentionModelBERT, 
    tokenizer: AutoTokenizer, 
    device: torch.device,
    turn_batch_size: int = 16  # Process 16 turns at a time
) -> float:
    """
    Runs a single inference on a list of transcript turns.
    This is the function your backend will call for each new request.
    
    This function now includes turn-level batching to prevent OOM errors.
    """
    
    # 1. Preprocessing (Tokenization)
    # Tokenize ALL turns, but keep them as standard Python lists
    inputs = tokenizer(
        transcript_turns, 
        padding="max_length",
        truncation=True,
        return_tensors=None  # <-- Return lists, not tensors
    )
    
    # We will store the sentence embeddings from the encoder here
    all_sentence_embeddings = []

    # 2. Run Encoder in Batches
    # This loop processes the turns in small chunks to avoid OOM
    print(f"Total turns: {len(transcript_turns)}. Processing in chunks of {turn_batch_size}...")
    for i in range(0, len(transcript_turns), turn_batch_size):
        # Create a small batch of input_ids and attention_masks
        batch_input_ids = torch.tensor(
            inputs["input_ids"][i:i + turn_batch_size]
        ).to(device)
        batch_attention_mask = torch.tensor(
            inputs["attention_mask"][i:i + turn_batch_size]
        ).to(device)

        # Pass this *small batch* through the model's encoder
        model_output = model.encoder(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask
        )
        
        # Run mean pooling
        sentence_embeddings = model.mean_pooling(
            model_output, batch_attention_mask
        )
        
        # Add the resulting embeddings to our list
        all_sentence_embeddings.append(sentence_embeddings.cpu())

    # Combine all the processed turn embeddings into one big tensor
    # This is the `sentence_embeddings` tensor from your model.py
    sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0).to(device)
    
    
    # 3. Create the Hierarchical (LSTM) Batch
    # This is the logic from your model.py's forward method
    
    # text_lens must be [1] (for batch_size=1 interview)
    # and the value is the total number of turns
    text_lens = torch.tensor([len(transcript_turns)]).to(device)
    
    # Split and pad (replicates your model's logic)
    word_outputs = torch.split(sentence_embeddings, text_lens.tolist())
    output = torch.nn.utils.rnn.pad_sequence(
        word_outputs, batch_first=True, padding_value=0
    )

    # 4. Run the Hierarchical Encoder (LSTM)
    batch_size = text_lens.size(0)
    sent_h0, sent_c0 = model.init_hidden(batch_size, device=device)
    
    # Call the SentenceLevelEncoder (the LSTM/Attention part)
    # Note: send_to_device is from accelerate, but text_lens is already on device.
    # We use .cpu() as per your original model.py code.
    (
        final_hidden_binary,
        final_hidden_regression,
        attn_binary,
        attn_regression,
        sent_conicity,
    ) = model.sent_encoder(
        output,
        text_lens.cpu(),
        sent_h0,
        sent_c0,
        device,
        pooling=model.pooling,
    )

    # 5. Run the Classification Head
    pred_binary = model.clf_binary(final_hidden_binary)
    
    # This is the 'symptom_scores_tensor'
    symptom_scores_tensor = pred_binary

    # 6. Sum the 8 scores to get the total
    total_score_tensor = torch.sum(symptom_scores_tensor, dim=1)
    
    # 7. Format and Return the Final Value
    final_score = total_score_tensor.squeeze().cpu().item()
    
    return final_score


# --- Main Execution Block ---
# This block runs ONLY when you execute the script directly 
# (e.g., `python inference_service.py`).
# It's an example of how to use the functions.
if __name__ == "__main__":
    
    """
    In a real backend service (like FastAPI or Flask), you would:
    1. Call set_device() and load_artifacts() *ONCE* when the server starts.
    2. Call get_depression_score() *inside your API endpoint* every time you get a new request.
    """
    
    # 1. Set the device
    device = set_device()
    
    try:
        # 2. Load the model and tokenizer (this can take a few seconds)
        print("Initializing service...")
        model, tokenizer = load_artifacts(
            model_path=MODEL_PATH,
            tokenizer_name=TOKENIZER_NAME,
            device=device
        )
        
        # 3. Simulate a new input (this would come from your API request)
        # This input MUST match the 'chunking' you used in training.
        # If CHUNKING was 'lines', this is just a list of lines.
        new_transcript = ["<sync>", "hi i'm ellie thanks for coming in today i was created to talk to people in a safe and secure environment i'm not a therapist but i'm here to learn about people and would love to learn about you i'll ask a few questions to get us started and please feel free to tell me anything your answers are totally confidential are you ok with this", "hello", " yes", "okay", "so how are you doing today", " okay", "not", "too bad", "that's good", "where are you from originally", "i'm from chicago", "south side", "when did you move to l_a", "um back in nineteen seventy eight", "when i was eighteen", "that's a long time", "yes <y>", "how often do you go back to your hometown", "lately", "not much maybe once every two or three years", "mhm", "how does it compare to l_a", "um it's totally different you know", "horrible weather and uh", "yeah it's just a whole different atmosphere", "why did you move to l_a", "um", "let's see my mother had moved to africa", "i graduated from high school", "and i had a friend out here so i just thought well let me try l_a", "okay", "how easy was it for you to get used to living in l_a", "uh very easy actually yeah i loved it from the moment i got off the airplane", "yeah", "what are some things you really like about l_a", "number one the weather", "um sun is very important to me", "so i love the weather", "um i don't know just kind of a more laid back atmosphere", "variety", "of people", "uh huh", "and now it's my um", "my home base you know", "my", "friends", "that are family", "so it's kinda like my family is here now", "what are some things you don't really like about l_a", "hmm", "i guess the smog earthquakes", "um", "traffic", "just basic things like that really", "yeah", "i'm sure you can tell by my shoes i'm not much of a world explorer do you travel a lot", "um i don't currently but i did throughout my life i traveled", "quite a lot", "what do you enjoy about traveling", "uh just seeing", "seeing", "other cultures and", "seeing um more history and", "meeting  people and", "just fun and the the fun of it as well", "mhm", "i'd love to hear about one of your trips", "um let's see well one trip", "i went to", "england", "ireland and scotland and france all in", "you know", "one trip", "so that was a nice variety", "nice", "what's one of your most memorable experiences", "um", "i guess you could say probably skiing", "in val d'isere france", "that was fun", "i'm glad i got to do that", "that's great", "do you consider yourself more shy or outgoing", "i'm kind of shy by nature", "i've tried to grow out of being shy so", "i'm not fully shy but i was always a an extremely shy", "child", "but now here in culver city this is my", "environment and i i guess you could say i'm not really shy in", "in my hometown of culver city", "yeah", "sometimes when i'm feeling tense i turn on the fish tank screensaver hey i know it's not hawaii but it's the best i've got what do you do to relax", "hm", "to relax i like to just curl up in my bed and read my book take a nap", "watch t_v", "sometimes on the couch in the living room", "just you know stay at home and", "kinda vegetate", "mhm", "how are you at controlling your temper", "uh pretty good yeah my temper's pretty level", "when was the last time you argued with someone and what was it about", "hm", "probably with my", "husband", "and", "i can't remember at the moment what it was about", "okay", "tell me about your relationship with your family", "um well when you say my family do you mean", "my  family from childhood or my current family", "whatever comes to your mind", "okay well i guess my current family comes to mind i i have two daughters", "and", "my relationship's pretty good it's okay with them but they're teenagers which uh", "you know", "going they're going through a lot of hormones and", "um", "they can be kinda bitchy <laughter> and", "(laughter)", "and um but i have hoped that you know they'll grow out of that and soon someday we'll have a really good relationship", "um and then there's the husband which", "i i guess i could say soon to be ex husband hopefully soon  <laughter>", "we need to get a divorce but we're still together", "due to financial reasons", "and yeah", "okay", "yeah so that that's not good the", "the relationship with the husband but um", "but since we have determined that we need to divorce", "our our we don't expect anything out out of each other so we don't argue as much so that's good", "yeah", "tell me about a situation that you wish you had handled differently", "hm", "well", "i guess", "our marriage", "i wish i had noticed early on that", "you know it was like way off from what i thought it was", "tell me about the hardest decision you've ever had to make", "um <sigh> let's see oh my god", "hardest decision", "i guess leaving my first husband that was pretty hard", "can you tell me about that", "um well you know we were married and i wasn't", "content i guess you could say", "um and", "i kinda just", "gave up on our relationship and", "i i now that i look back i think it's because", "i wanted", "children and he would've been happy to never have any", "and so uh since i didn't have a father", "it was very important", "that i", "create children with someone who really wanted to be a father", "and so i was kind of", "at a age where i needed to start thinking about", "having children", "yet i was with someone who really was not into it so that", "i think that's one uh maybe the core reason why i was not content and i just", "felt like", "<tisk> this marriage isn't gonna work and", "so you know we went to therapy and everything and he did not wanna break up and", "<tisk> um", "<tisk> and i had to leave him", "because i just wasn't happy and that was that was very  hard 'cause", "you know", "i felt like i was hurting him  which which i was you know and he survived he got over it now he's  married with two kids <laughter>", "<laughter> but um", "it was", "hard", "to leave him", "yeah", "tell me about an event or something that you wish you could erase from your memory", "mm", "something i wish i could erase from my memory", "i have to think about that", "i'm sure there's something <gasp> oh well", "yeah there's um", "i guess it'd be nice to erase", "from my memory  um a recent i mean not recent but an event", "in my", "recent adulthood here in culver city", "where one of my best friends", "um", "i've betrayed me by", "back years ago when my husband first got into real estate", "she and her family were growing and they really needed  a larger home", "and", "um", "we just assumed that you know that they would use", "my husband as their agent both to sell their current home current at the time", "and purchase a new home", "<tisk> and as it turned out", "they surprised us and", "had been talking to this other local agent", "and uh yeah", "that was devastating", "i don't think i had ever  been betrayed", "like that", "so i wish i could  erase that now 'cause now you know this was back when the kids were really young", "and um", "my daughter at the time my youngest daughter was let's say", "six maybe", "um", "maybe five", "anyway her <h>", "her and my best friend's", "daughter are the same age they were born within days of each other", "we were pregnant together", "and um", "they were really tight those two little girls and they ended up not going to the same elementary school  so they didn't get to hang out during those years but every time we'd run into them in the neighborhood or they'd", "run into each other at a summer camp or something they were just so connected they were", "just you know like two peas in a pod even though it's her", "their moms weren't talking to each other anymore", "so now they're thirteen and fourteen and they're in middle school together", "and they're still very connected in fact <laughter>", "she she for the first time spent the night at our house last night with my daughter", "and another girl", "and it's just a little awkward you know 'cause now", "these girls are", "becoming really good friends better friends", "um", "and yet", "i still have this", "vivid", "um", "bad memory", "of what her", "her mom did to", "to me", "yeah", "that was about what", "ten", "about eight years ago", "anyway", "<sigh> i tried to i tried to", "i'm sorry", "ignore it and put it out of my head and", "which i've been able to <sniff>", "in recent years", "but obviously it still comes up", "yeah", "anyway that that was that", "how easy is it for you to get a good night's sleep", "<sniff>", "um not that easy", "i do sometimes get a decent night's sleep but", "sometimes it's hard to get to sleep and most of the time i wake up", "and can't get back to sleep", "uh or i finally get back to sleep and then it's almost time to get up you know that", "that difficult thing", "are they triggered by something", "well the waking up in the middle of the night is definitely triggered by having to go to the bathroom <laughter>", "and then um", "<tisk> you know it's not always that easy to get back to sleep and then i wake up again to go to the bathroom", "so i guess that's the trigger", "as far as getting to sleep having a hard time there", "i don't know i think there's just too much on my mind", "running through my mind and the day", "i feel like i haven't accomplished enough during the day so", "it's hard for me to just go to sleep when i'm supposed to", "i guess", "okay", "what are you like when you don't sleep well", "um", "well i'm kinda used to it now", "so i just go on", "through the day probably looking <laughter> looking tired and um", "i i'm kinda used to feeling tired i just do what i need to do", "and sometimes if i'm able to if i'm at home and don't have", "huge commitments yeah i might just like i said earlier just take a nap or something", "read my book watch t_v watch whatever", "mhm", "how have you been feeling lately", "lately how have i been feeling um", "kind of down", "on myself down on life", "can you tell me about that", "well i have this pending", "need <n> need for separation slash divorce", "um", "<tisk> i worry that i made a lot of mistakes with my girls now they're teenagers", "you know on the verge of adulthood", "<clears throat>", "and so i worry that", "you know", "i did all kinds of things wrong", "with their guidance and their early childhood and  <clears throat> now it's too late", "um", "like i missed out on a thing <th> a lot of things", "uh", "and yeah i need a job i'm unemployed i don't know what i wanna do with my life", "i know i wanna do something i i", "i just wish", "you know i never did finish college unfortunately because i was traveling and stuff", "so i i regret that extremely", "and i need to get a job and be able to take care of myself 'cause i need to", "divorce and move on with my life", "so", "now i forget what the original question was but  <laughter>", "um", "that's the state", "part of the state of my life", "okay", "have you ever been diagnosed with p_t_s_d", "no", "have you been diagnosed with depression", "yes", "how long ago were you diagnosed", "mm <tisk>", "well", "the first time or or or recently", "yes", "the first time i guess was", "you know when i was in my early twenties i was just a little", "upset about <laughter> coincidentally that was another breakup i had to make back then", "i was um", "<tisk> in my early twenties and i needed to break up with my boyfriend but again it was very upsetting 'cause i", "you know it was hard because he didn't wanna break up and i felt like i was hurting him", "and i didn't understand this feeling", "i was really upset", "how i didn't that was my first time having depression so i didn't really", "recognize <r> recognize it as depression but i went somehow i", "had the", "knowledge or the", "i was able to", "take myself to a psychiatrist i remember i paid for her by the hour", "and she prescribed xanax to just help relax me", "and so i guess that helped me get through that and then i've had depression throughout the years just you know little bits here and there for various reasons such as", "i guess it's usually a relationship <laughter> you know", "um", "yeah so the most recent time i was diagnosed", "and put on meds", "that was probably about", "two years ago", "mm", "do you still go to therapy now", "no", "why did you stop", "um money i i don't have insurance", "otherwise yeah i'd be happy to", "go to therapy <laughter>", "anything that might help", "tell me about the last time you felt really happy", "ooh <laughter>", "boy", "last time i felt really happy", "um", "i don't know i guess maybe", "when my children were born", "that's", "how did you feel in that moment", "<laughter>", "um", "i guess happy i don't know <laughter> i just you know", "i always wanted children so it was nice", "to have them", "that's", "i don't know that's all i can remember of that i don't i guess i don't remember the good times too well <laughter> i remember the bad times", "yeah", "tell me about something you did recently that you really enjoyed", "<sniff>", "mm", "something i did that i really enjoyed", "well", "let's see i um", "friday night i went to a friend's house", "and met up with", "five other girlfriends and we", "we created vision boards", "and talked and", "had dinner", "at one of their house that was that was fun", "that sounds like a great situation", "yeah", "we get to", "what advice would you give to yourself ten or twenty years ago", "ooh <sigh>", "stay in college get my degree", "um", "get therapy try to become a happy", "um", "yeah try to figure out things earlier in life", "like", "yeah", "stay in college and oh and mainly i think the main you know what it all boils down to <laughter> no matter what you're depressed about it seems like", "as as the saying goes and the songs <s> the", "songs go you know", "it's all about money the world revolves around money and", "having none", "that in itself is very depressing <laughter> so besides relationships and different things  that have happened in your life", "having no money", "is", "you know it's like you can't even try to fix anything because you don't have the money to fix it", "so", "with that in mind i would uh definitely advise myself to get my college degree", "and to stick <s> get a job and stick with it", "um", "don't quit", "to go traveling you know", "just stick with a job stick with a company build up your", "seniority keep your benefits", "and build up some knowledge or some expertise in some sort of field", "yeah that would've been", "my biggest piece of advice to myself", "okay", "what are you most proud of in your life", "not much <laughter>", "i guess", "just my daughters you know they're", "beautiful daughters and", "i'm proud and glad that i was able to have children some people", "aren't even able to do that you know and they  try so hard or they look back on their life and they wish they had done it so", "that's the one thing i mean i wanted", "five children and i always wanted a boy which i never had but", "um i'm", "i guess that's the", "biggest thing i can", "think of to be proud of", "is having my two daughters", "what would you say are some of your best qualities", "um", "i'm a very loyal", "friend and person", "that's why", "when that friend betrayed me it was so devastating", "yeah very loyal um", "very", "<inhale> happy <hap> i mean huh what was i gonna say i was gonna say happy lucky <l> go lucky i don't know where that came from i'm very i meant to say friendly", "and", "caring and", "um yeah i care about people", "and i'm interested in people", "and", "i care about", "animals as well", "actually that's another thing i'm proud of proud of being a vegetarian", "yeah so my daughters and being vegetarian", "i'm a vegetarian because i really care about", "feelings of other beings not only humans but animals too and", "you know i don't think that animals should be  tortured", "and live a a tortuous life", "just so we can eat 'em", "yeah", "so um", "what was the question", "do you remember the question", "okay i think i have asked everything i need to", "okay", "thanks for sharing your thoughts with me", "you're welcome", "goodbye", "bye"]
        # 4. Get the prediction
        print("\nRunning inference on sample transcript...")
        total_phq_score = get_depression_score(
            new_transcript, 
            model, 
            tokenizer, 
            device
        )

        print("\n--- INFERENCE COMPLETE ---")
        print(f"Predicted Total Depression Score: {total_phq_score:.4f}")

    except Exception as e:
        print(f"\nAn error occurred during initialization or inference:")
        print(f"{e}")
        print("Please check your file paths and dependencies.")