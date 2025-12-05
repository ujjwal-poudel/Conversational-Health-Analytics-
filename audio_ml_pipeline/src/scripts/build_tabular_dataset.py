"""
Build Tabular Dataset for Depression Detection

Experiment versions:
- V5: PCA only (768→50), no segment pooling
- V6: PCA (768→100) + Segment pooling (3 segments) - ACTIVE

Toggle between versions by commenting/uncommenting the appropriate sections.
"""

import os
import numpy as np
import pandas as pd

from src.utils import config
from src.utils.logging_utils import get_logger
from src.utils.pooling import pool_embeddings
from src.utils.pca_utils import fit_pca_on_embeddings, apply_pca_to_embedding

logger = get_logger(__name__)


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# V6: PCA + Segment Pooling Hybrid (ACTIVE)
PCA_N_COMPONENTS = 200  # Reduce 768 -> 200 (preserves 96.3% variance)
USE_SEGMENT_POOLING = True  # Enable segment pooling

# V5 settings (commented out):
# PCA_N_COMPONENTS = 50  # V5: Reduce 768 -> 50
# USE_SEGMENT_POOLING = False  # V5: No segment pooling


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_embedding(path):
    if not os.path.exists(path):
        return None
    return np.load(path)


# =============================================================================
# V6: PCA + SEGMENT POOLING HYBRID (ACTIVE)
# =============================================================================

def build_feature_vector_v6(pid, pca_model):
    """
    V6: PCA + Segment Pooling Hybrid.
    
    Steps:
    1. Load Wav2Vec2 (768 dims) and Prosody (13 dims)
    2. Apply PCA to Wav2Vec2 (768 -> 100)
    3. Apply segment pooling (3 segments × 4 stats)
    
    Expected output:
    - Wav2Vec2: 100 × 4 × 3 = 1,200 features
    - Prosody:  13 × 4 × 3 = 156 features
    - Total: 1,356 features
    """
    wav2vec_path = os.path.join(config.FEATURE_WAV2VEC2_DIR, f"{pid}_P.npy")
    prosody_path = os.path.join(config.FEATURE_PROSODY_DIR, f"{pid}_P.npy")

    emb_w = load_embedding(wav2vec_path)
    emb_p = load_embedding(prosody_path)

    if emb_w is None or emb_p is None:
        logger.warning(f"Missing embeddings for participant {pid}")
        return None

    # Step 1: Apply PCA to Wav2Vec2 (768 -> 100)
    emb_w_reduced = apply_pca_to_embedding(emb_w, pca_model)

    # Step 2: Apply segment pooling (3 segments × 4 stats)
    pooled_w = pool_embeddings(emb_w_reduced, use_segments=USE_SEGMENT_POOLING)
    pooled_p = pool_embeddings(emb_p, use_segments=USE_SEGMENT_POOLING)

    # Step 3: Concatenate
    combined = np.concatenate([pooled_w, pooled_p])
    return pooled_w, pooled_p, combined


# =============================================================================
# V5: PCA ONLY (COMMENTED OUT - keep for inference reference)
# =============================================================================

# def build_feature_vector_v5(pid, pca_model):
#     """
#     V5: PCA only, no segment pooling.
#     
#     Steps:
#     1. Load Wav2Vec2 (768 dims) and Prosody (13 dims)
#     2. Apply PCA to Wav2Vec2 (768 -> 50)
#     3. Apply simple pooling (4 stats only)
#     
#     Expected output:
#     - Wav2Vec2: 50 × 4 = 200 features
#     - Prosody:  13 × 4 = 52 features
#     - Total: 252 features
#     """
#     wav2vec_path = os.path.join(config.FEATURE_WAV2VEC2_DIR, f"{pid}_P.npy")
#     prosody_path = os.path.join(config.FEATURE_PROSODY_DIR, f"{pid}_P.npy")
#
#     emb_w = load_embedding(wav2vec_path)
#     emb_p = load_embedding(prosody_path)
#
#     if emb_w is None or emb_p is None:
#         logger.warning(f"Missing embeddings for participant {pid}")
#         return None
#
#     # Apply PCA to wav2vec2 (768 -> 50)
#     emb_w_reduced = apply_pca_to_embedding(emb_w, pca_model)
#
#     # Pool (no segment pooling)
#     pooled_w = pool_embeddings(emb_w_reduced, use_segments=False)
#     pooled_p = pool_embeddings(emb_p, use_segments=False)
#
#     combined = np.concatenate([pooled_w, pooled_p])
#     return pooled_w, pooled_p, combined


# =============================================================================
# V4: SEGMENT POOLING ONLY (COMMENTED OUT - keep for inference reference)
# =============================================================================

# def build_feature_vector_v4(pid):
#     """
#     V4: Segment pooling without PCA (original best model).
#     
#     Expected output:
#     - Wav2Vec2: 768 × 4 × 3 = 9,216 features
#     - Prosody:  13 × 4 × 3 = 156 features
#     - Total: 9,372 features
#     """
#     wav2vec_path = os.path.join(config.FEATURE_WAV2VEC2_DIR, f"{pid}_P.npy")
#     prosody_path = os.path.join(config.FEATURE_PROSODY_DIR, f"{pid}_P.npy")
#
#     emb_w = load_embedding(wav2vec_path)
#     emb_p = load_embedding(prosody_path)
#
#     if emb_w is None or emb_p is None:
#         logger.warning(f"Missing embeddings for participant {pid}")
#         return None
#
#     # Segment pooling (3 segments × 4 stats)
#     pooled_w = pool_embeddings(emb_w, use_segments=True)
#     pooled_p = pool_embeddings(emb_p, use_segments=True)
#
#     combined = np.concatenate([pooled_w, pooled_p])
#     return pooled_w, pooled_p, combined


# =============================================================================
# SAVE AND BUILD FUNCTIONS
# =============================================================================

def save_individual(pid, pooled_w, pooled_p, pooled_all, label):
    """Saves pooled features + label for a single participant."""
    root = config.MODEL_DIR

    out_w = os.path.join(root, "pooled_wav2vec2")
    out_p = os.path.join(root, "pooled_prosody")
    out_a = os.path.join(root, "pooled_all")

    ensure_dir(out_w)
    ensure_dir(out_p)
    ensure_dir(out_a)

    np.save(os.path.join(out_w, f"{pid}.npy"), np.append(pooled_w, label))
    np.save(os.path.join(out_p, f"{pid}.npy"), np.append(pooled_p, label))
    np.save(os.path.join(out_a, f"{pid}.npy"), np.append(pooled_all, label))


def build_split_with_pca(split_df, save_path, pca_model):
    """Builds the dataset for a single split using PCA + segment pooling."""
    rows = []
    total = len(split_df)

    for idx, (_, row) in enumerate(split_df.iterrows()):
        pid = str(row["participant_id"])
        label = float(row["PHQ_score"])

        logger.info(f"Processing {idx+1}/{total}: participant {pid}")

        # V6: Use PCA + Segment Pooling
        pooled = build_feature_vector_v6(pid, pca_model)
        
        # V5: Use PCA only (commented out)
        # pooled = build_feature_vector_v5(pid, pca_model)
        
        # V4: Use segment pooling only (commented out)
        # pooled = build_feature_vector_v4(pid)

        if pooled is None:
            continue

        pooled_w, pooled_p, pooled_all = pooled
        save_individual(pid, pooled_w, pooled_p, pooled_all, label)

        row_vector = np.append(pooled_all, label)
        rows.append(row_vector)

    if len(rows) == 0:
        logger.error("No valid participants found in this split.")
        return

    data = np.vstack(rows)
    np.save(save_path, data)
    logger.info(f"Saved dataset to {save_path}  Shape: {data.shape}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_build_tabular_dataset():
    """
    V6: PCA + Segment Pooling Hybrid Experiment.
    
    Pipeline:
    1. Load all training Wav2Vec2 embeddings
    2. Fit PCA (768 -> 100) on training data
    3. For each participant:
       - Apply PCA to Wav2Vec2
       - Apply segment pooling (3 segments × 4 stats)
    4. Save train/dev/test datasets
    
    Expected features per participant:
    - Wav2Vec2: 100 dims × 4 stats × 3 segments = 1,200
    - Prosody:  13 dims × 4 stats × 3 segments = 156
    - Total: 1,356 features
    """

    train_df = pd.read_csv(config.TRAIN_LABEL_FILE)
    dev_df = pd.read_csv(config.DEV_LABEL_FILE)
    test_df = pd.read_csv(config.TEST_LABEL_FILE)

    test_df = test_df.rename(columns={"PHQ_Score": "PHQ_score"})

    train_df["participant_id"] = train_df["Participant_ID"].astype(str)
    dev_df["participant_id"] = dev_df["Participant_ID"].astype(str)
    test_df["participant_id"] = test_df["Participant_ID"].astype(str)

    train_df["PHQ_score"] = train_df["PHQ8_Score"]
    dev_df["PHQ_score"] = dev_df["PHQ8_Score"]

    # Step 1: Load all wav2vec2 embeddings from training set
    logger.info("=" * 60)
    logger.info("V6: PCA + Segment Pooling Hybrid")
    logger.info("=" * 60)
    logger.info("Loading training embeddings for PCA fitting...")
    
    train_embeddings = []
    for pid in train_df["participant_id"]:
        wav2vec_path = os.path.join(config.FEATURE_WAV2VEC2_DIR, f"{pid}_P.npy")
        emb = load_embedding(wav2vec_path)
        if emb is not None:
            train_embeddings.append(emb)

    # Step 2: Fit PCA on training embeddings
    logger.info(f"Fitting PCA: 768 -> {PCA_N_COMPONENTS} dimensions...")
    pca_model = fit_pca_on_embeddings(train_embeddings, n_components=PCA_N_COMPONENTS)

    # Step 2b: Save PCA for inference
    from src.utils.pca_utils import save_pca
    pca_save_path = os.path.join(config.MODEL_DIR, "pca_wav2vec2.joblib")
    save_pca(pca_model, pca_save_path)

    # Step 3: Build splits
    train_out = os.path.join(config.MODEL_DIR, "train_data.npy")
    dev_out = os.path.join(config.MODEL_DIR, "dev_data.npy")
    test_out = os.path.join(config.MODEL_DIR, "test_data.npy")

    logger.info("Building TRAIN dataset")
    build_split_with_pca(train_df, train_out, pca_model)

    logger.info("Building DEV dataset")
    build_split_with_pca(dev_df, dev_out, pca_model)

    logger.info("Building TEST dataset")
    build_split_with_pca(test_df, test_out, pca_model)

    # Calculate expected features
    if USE_SEGMENT_POOLING:
        n_stats = 4
        n_segments = 3
        wav2vec_features = PCA_N_COMPONENTS * n_stats * n_segments
        prosody_features = 13 * n_stats * n_segments
    else:
        n_stats = 4
        wav2vec_features = PCA_N_COMPONENTS * n_stats
        prosody_features = 13 * n_stats

    total_features = wav2vec_features + prosody_features

    logger.info("=" * 60)
    logger.info("Dataset building complete!")
    logger.info(f"Wav2Vec2 features: {wav2vec_features}")
    logger.info(f"Prosody features: {prosody_features}")
    logger.info(f"Total features: {total_features}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_build_tabular_dataset()