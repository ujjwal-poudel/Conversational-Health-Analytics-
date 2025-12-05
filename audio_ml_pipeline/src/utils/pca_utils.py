"""
PCA utilities for dimensionality reduction of embeddings.

Reduces high-dimensional embeddings (e.g., 768-dim Wav2Vec2) to 
a smaller subspace before pooling, to prevent overfitting.

IMPORTANT: For inference, you must load the same fitted PCA object
that was used during training. Use save_pca() and load_pca().
"""

import numpy as np
import joblib
from sklearn.decomposition import PCA


def fit_pca_on_embeddings(embedding_list, n_components=50):
    """
    Fits PCA on all embeddings from training set.
    
    Parameters
    ----------
    embedding_list : list of np.ndarray
        List of (T_i, D) embedding matrices from training participants.
    n_components : int
        Number of PCA components to keep.
    
    Returns
    -------
    PCA
        Fitted PCA object.
    """
    # Stack all frames from all participants
    all_frames = np.vstack(embedding_list)
    
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(all_frames)
    
    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA fitted: {n_components} components explain {explained_var:.1f}% variance")
    
    return pca


def apply_pca_to_embedding(embedding, pca):
    """
    Applies fitted PCA to a single embedding matrix.
    
    Parameters
    ----------
    embedding : np.ndarray
        Embedding matrix of shape (T, D).
    pca : PCA
        Fitted PCA object.
    
    Returns
    -------
    np.ndarray
        Reduced embedding of shape (T, n_components).
    """
    return pca.transform(embedding)


def save_pca(pca, path):
    """
    Saves fitted PCA object for inference.
    
    Parameters
    ----------
    pca : PCA
        Fitted PCA object.
    path : str
        Path to save the PCA object (e.g., 'models/pca_wav2vec2.joblib')
    """
    joblib.dump(pca, path)
    print(f"PCA saved to: {path}")


def load_pca(path):
    """
    Loads fitted PCA object for inference.
    
    Parameters
    ----------
    path : str
        Path to the saved PCA object.
    
    Returns
    -------
    PCA
        Fitted PCA object.
    """
    pca = joblib.load(path)
    print(f"PCA loaded from: {path} ({pca.n_components_} components)")
    return pca
