"""
Configuration module for the audio ML pipeline.

Defines:
- Paths for DAIC original audio and transcripts
- Paths for cleaned audio, feature outputs, summary features, and model files
- Preprocessing parameters
- Feature extraction parameters (paper settings)
- Model training settings
"""

import os


# External folder created by the user to store all pipeline outputs
AUDIO_DATA_ROOT = "/Volumes/MACBACKUP/audio_data_folder"


# Processed directories for pipeline outputs
PROCESSED_AUDIO_DIR = os.path.join(AUDIO_DATA_ROOT, "processed")
FEATURE_DIR = os.path.join(AUDIO_DATA_ROOT, "features")
SUMMARY_FEATURE_DIR = os.path.join(AUDIO_DATA_ROOT, "summary")
MODEL_DIR = os.path.join(AUDIO_DATA_ROOT, "models")

# Subfolders for each feature type
FEATURE_MFCC_DIR = os.path.join(FEATURE_DIR, "mfcc")
FEATURE_CHROMA_DIR = os.path.join(FEATURE_DIR, "chroma")
FEATURE_SPECTRAL_DIR = os.path.join(FEATURE_DIR, "spectral")
FEATURE_TONNETZ_DIR = os.path.join(FEATURE_DIR, "tonnetz")
FEATURE_RMS_DIR = os.path.join(FEATURE_DIR, "rms")
FEATURE_ZCR_DIR = os.path.join(FEATURE_DIR, "zcr")
FEATURE_COMBINED_DIR = os.path.join(FEATURE_DIR, "combined")

FEATURE_WAV2VEC2_DIR = os.path.join(FEATURE_DIR, "wav2vec2")
FEATURE_OPENL3_DIR = os.path.join(FEATURE_DIR, "openl3")
FEATURE_PROSODY_DIR = os.path.join(FEATURE_DIR, "prosody")

# Create directories if missing
for d in [
    PROCESSED_AUDIO_DIR,
    FEATURE_DIR,
    SUMMARY_FEATURE_DIR,
    MODEL_DIR,
    FEATURE_MFCC_DIR,
    FEATURE_CHROMA_DIR,
    FEATURE_SPECTRAL_DIR,
    FEATURE_TONNETZ_DIR,
    FEATURE_RMS_DIR,
    FEATURE_ZCR_DIR,
    FEATURE_COMBINED_DIR,
    FEATURE_WAV2VEC2_DIR,
    FEATURE_OPENL3_DIR,
    FEATURE_PROSODY_DIR,
]:
    os.makedirs(d, exist_ok=True)


# Original DAIC-WOZ dataset location
DAIC_ORIGINAL_ROOT = "/Volumes/MACBACKUP/extracted_folders"


# Participant ID range
PARTICIPANT_MIN_ID = 300
PARTICIPANT_MAX_ID = 492


# Label files
TRAIN_LABEL_FILE = "/Volumes/MACBACKUP/train_split_Depression_AVEC2017.csv"
DEV_LABEL_FILE = "/Volumes/MACBACKUP/dev_split_Depression_AVEC2017.csv"
TEST_LABEL_FILE = "/Volumes/MACBACKUP/full_test_split.csv"


# Audio preprocessing parameters
AUDIO_SR = 16000
HIGHPASS_CUTOFF = 300
HIGHPASS_ORDER = 5

ELLIE_LABELS = ["Ellie", "Therapist"]
PARTICIPANT_LABEL = "Participant"


# Feature extraction parameters (paper settings)
WINDOW_LENGTH = 4096
HOP_LENGTH = 2205

MFCC_N_MFCC = 80
CHROMA_N_CHROMA = 12

SUMMARY_STATS = ["mean", "std", "min", "max"]


# Enabled feature types
FEATURES_ENABLED = {
    "mfcc": True,
    "chroma": True,
    "spectral": True,
    "tonnetz": True,
    "rms": True,
    "zcr": True,
}


# Modeling
MODEL_TYPE = "lightgbm"
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.2

FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "final_model.pkl")
FINAL_FEATURE_LIST = os.path.join(MODEL_DIR, "final_feature_list.json")


def build_raw_audio_path(participant_id):
    """
    Construct the original DAIC-WOZ audio file path.
    """
    # Extract numeric ID (e.g., "300" from "300_P")
    numeric_id = participant_id.replace("_P", "")
    folder = os.path.join(DAIC_ORIGINAL_ROOT, participant_id)
    filename = f"{numeric_id}_AUDIO.wav"
    return os.path.join(folder, filename)


def build_transcript_path(participant_id):
    """
    Construct the transcript file path for a participant.
    """
    # Extract numeric ID (e.g., "300" from "300_P")
    numeric_id = participant_id.replace("_P", "")
    folder = os.path.join(DAIC_ORIGINAL_ROOT, participant_id)
    filename = f"{numeric_id}_TRANSCRIPT.csv"
    return os.path.join(folder, filename)


def get_processed_audio_path(participant_id):
    """
    Path for cleaned participant-only audio.
    """
    return os.path.join(PROCESSED_AUDIO_DIR, f"{participant_id}.wav")


def get_summary_feature_path(participant_id):
    """
    Path for storing aggregated summary feature CSV.
    """
    return os.path.join(SUMMARY_FEATURE_DIR, f"{participant_id}.csv")


def get_model_path(filename="final_model.pkl"):
    """
    Path for storing or loading trained ML models.
    """
    return os.path.join(MODEL_DIR, filename)