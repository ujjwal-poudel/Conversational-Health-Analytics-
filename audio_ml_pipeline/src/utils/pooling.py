"""
Pooling utilities for converting variable-length embeddings (T, D)
into fixed-length summary vectors suitable for XGBoost / LightGBM.

Supported pooling operations (reduced to 4 for better generalization):
- mean
- std
- min
- max

# Previously used 8 stats - commented out to reduce dimensions:
# - median
# - 25th percentile
# - 75th percentile
# - skew

Optional segment pooling is available: beginning, middle, end.
"""

import numpy as np
# from scipy.stats import skew  # Not needed with reduced stats


def compute_summary_stats(matrix):
    """
    Computes statistical summary features for a (T, D) matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Embedding matrix of shape (T, D).

    Returns
    -------
    np.ndarray
        Flattened summary vector of shape (D * num_stats,)
    """
    # (T, D)
    X = matrix

    # Reduced to 4 core statistics for better generalization
    # This reduces feature count from ~6248 to ~3124 for wav2vec2+prosody
    stats = []
    stats.append(np.mean(X, axis=0))
    stats.append(np.std(X, axis=0))
    stats.append(np.min(X, axis=0))
    stats.append(np.max(X, axis=0))
    
    # Commented out to reduce dimensions (uncomment for full 8 stats):
    # stats.append(np.median(X, axis=0))
    # stats.append(np.percentile(X, 25, axis=0))
    # stats.append(np.percentile(X, 75, axis=0))
    # stats.append(skew(X, axis=0, bias=False))

    pooled = np.concatenate(stats)

    # Remove any NaN values safely
    pooled = np.nan_to_num(pooled)

    return pooled


def segment_pool(matrix, segments=3):
    """
    Applies summary stats pooling to each segment of the embedding matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Embedding matrix of shape (T, D)
    segments : int
        Number of segments to split the matrix into (default = 3)

    Returns
    -------
    np.ndarray
        Concatenated pooled vector from all segments.
        Shape: (segments * D * num_stats,)
    """
    T = matrix.shape[0]
    splits = np.array_split(matrix, segments)

    pooled_segments = [compute_summary_stats(seg) for seg in splits]

    return np.concatenate(pooled_segments)


def pool_embeddings(matrix, use_segments=False):
    """
    Main pooling function.

    Parameters
    ----------
    matrix : np.ndarray
        Embedding matrix of shape (T, D)
    use_segments : bool
        If True, performs segment pooling (begin/middle/end).
        If False, simple summary stats over the full sequence.

    Returns
    -------
    np.ndarray
        Pooled feature vector.
    """
    if use_segments:
        return segment_pool(matrix, segments=3)

    return compute_summary_stats(matrix)