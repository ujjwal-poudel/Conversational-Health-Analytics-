from typing import List, Tuple
from src.semantic_inference import get_semantic_classifier

def analyze_with_semantic_guardrails(
    turns: List[str], 
    regression_score: float
) -> Tuple[float, str, str, float, float]:
    """
    Combines the primary Regression Model score with a secondary Semantic Similarity analysis
    to act as a safety guardrail.

    Args:
        turns (List[str]): The list of conversation turns (questions and answers).
        regression_score (float): The raw score output from the main regression model.

    Returns:
        Tuple containing:
        1. Final Score (float): The score to show the user (usually the regression score).
        2. Semantic Label (str): "High Risk" or "Low Risk" based on semantic similarity.
        3. Consistency Status (str): A tag describing if models agreed or conflicted.
        4. Similarity to Healthy (float): Raw cosine similarity to the 'Healthy' prototype.
        5. Similarity to Depressed (float): Raw cosine similarity to the 'Depressed' prototype.
    """
    
    # --- STEP 1: Perform Semantic Analysis ---
    try:
        # Join the list of turns into one single string for embedding
        full_text = " ".join(turns)
        
        # Load the singleton semantic classifier
        classifier = get_semantic_classifier()
        
        # Get the prediction dictionary
        semantic_result = classifier.predict(full_text)
        
        # Extract the raw data points
        sim_0 = semantic_result["similarity_class_0"] # Similarity to 'Healthy' prototype
        sim_1 = semantic_result["similarity_class_1"] # Similarity to 'Depressed' prototype
        semantic_class = semantic_result["predicted_class"] # 0 for Healthy, 1 for Depressed
        label = semantic_result["predicted_label"] # Text label for display
        
        # Calculate "Confidence Margin":
        # If sim_0 is 0.45 and sim_1 is 0.46, the margin is 0.01 (Very Low Confidence).
        # If sim_0 is 0.20 and sim_1 is 0.70, the margin is 0.50 (Very High Confidence).
        confidence_margin = abs(sim_1 - sim_0)
        
    except Exception as e:
        # ERROR HANDLING: If the semantic check fails (e.g., file missing),
        # we do NOT stop the app. We fail safely by returning the regression score.
        print(f"Semantic Analysis Failed: {e}")
        return regression_score, "N/A", "semantic_failed", 0.0, 0.0

    # --- STEP 2: Standardize the Regression Output ---
    # We convert the continuous regression score (e.g., 8.5 or 12.3) into a binary class
    # using the standard PHQ-8 threshold of 10.
    regression_class = 1 if regression_score >= 10 else 0
    
    # Default status
    status = "processed"

    # --- STEP 3: The Logic Gates (Guardrails) ---
    
    # GATE 1: Low Confidence Check
    # If the semantic model is "unsure" (the text is ambiguous and equally similar to both),
    # we should NOT let it override or flag the main model.
    # We trust the Regression Model completely in this case.
    if confidence_margin < 0.05:
        status = "agreement_low_confidence"
        return regression_score, label, status, sim_0, sim_1

    # GATE 2: Agreement Check
    # Ideally, both models agree (e.g., both say "High Risk" or both say "Low Risk").
    # This validates our prediction and gives us high confidence.
    if semantic_class == regression_class:
        status = "strong_agreement"
        return regression_score, label, status, sim_0, sim_1

    # GATE 3: Handling Conflicts (The Models Disagree)
    
    # Case A: Semantic says DEPRESSED (1), but Regression says HEALTHY (0)
    # This is a potential FALSE NEGATIVE. 
    # The user's words are semantically close to "depressed" text, 
    # but the regression model calculated a low score (e.g., 8 or 9).
    if semantic_class == 1 and regression_class == 0:
        status = "conflict_potential_false_negative"
        # We return the Regression Score (because it's the primary clinical tool),
        # but the 'status' flag warns us to review this case manually or treat it with caution.
        return regression_score, "High Risk (Semantic)", status, sim_0, sim_1

    # Case B: Semantic says HEALTHY (0), but Regression says DEPRESSED (1)
    # This is a potential FALSE POSITIVE.
    # The user sounds "normal" semantically, but scored high on the clinical scale.
    # In medical contexts, we prefer False Positives over False Negatives (better safe than sorry).
    # We stick with the High Score to ensure the user gets help if needed.
    if semantic_class == 0 and regression_class == 1:
        status = "conflict_potential_false_positive"
        return regression_score, "Low Risk (Semantic)", status, sim_0, sim_1

    # Fallback return (should theoretically not be reached given logic above)
    return regression_score, label, status, sim_0, sim_1
