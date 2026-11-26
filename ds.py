"""
Ex-12: Clustering Taylor Swift Songs (Soft Clustering with GMM)

Goal:
- Implement Gaussian Mixture Models (GMM) to discover musical clusters
- Choose k (# components) using BIC (or AIC)
- Visualize clusters in 2D using PCA
- Interpret clusters via component means (centroids) in original feature space

Dataset:
- data/taylor_album_songs.csv

Your job in this file:
- Implement the TODO sections.
- Keep functions small and testable.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------
# Step 1: Load + Preprocess
# ---------------------------------------------------------------------

DEFAULT_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


def load_songs(path: str = "data/taylor_album_songs.csv") -> pd.DataFrame:
    """
    Load the Taylor Swift album songs dataset.

    TODO:
    - Read CSV
    - Return DataFrame
    """
    raise NotImplementedError("Complete load_songs() TODOs.")


def select_features(df: pd.DataFrame, features=None) -> pd.DataFrame:
    """
    Select numeric features for clustering.

    TODO:
    - Default to DEFAULT_FEATURES if features is None
    - Coerce selected columns to numeric (errors='coerce')
    - Drop rows with missing values in selected features
    - Return X as a DataFrame of selected features only
    """
    raise NotImplementedError("Complete select_features() TODOs.")


def scale_features(X: pd.DataFrame):
    """
    Standardize features (mean=0, std=1) before GMM.

    TODO:
    - Fit StandardScaler on X
    - Transform X
    - Return (X_scaled, scaler)
    """
    raise NotImplementedError("Complete scale_features() TODOs.")


# ---------------------------------------------------------------------
# Step 2: Fit GMM + Model Selection (BIC/AIC)
# ---------------------------------------------------------------------

def fit_gmm(X_scaled: np.ndarray, n_components: int, random_state: int = 42):
    """
    Fit a Gaussian Mixture Model.

    TODO:
    - Initialize GaussianMixture with:
        n_components=n_components
        covariance_type='full' (default is fine)
        random_state=random_state
    - Fit on X_scaled
    - Return fitted model
    """
    raise NotImplementedError("Complete fit_gmm() TODOs.")


def select_k_bic(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
):
    """
    Choose the number of components using BIC.

    TODO:
    - Fit GMMs for k in [k_min, k_max]
    - Compute BIC for each model
    - Return (best_k, results_df) where results_df has columns:
        ['k', 'bic']
    - best_k is the k with the smallest BIC
    """
    raise NotImplementedError("Complete select_k_bic() TODOs.")


# ---------------------------------------------------------------------
# Step 3: Soft assignments + Visualization (PCA)
# ---------------------------------------------------------------------

def soft_assignments(model: GaussianMixture, X_scaled: np.ndarray) -> np.ndarray:
    """
    Return soft cluster membership probabilities.

    TODO:
    - Use model.predict_proba(X_scaled)
    """
    raise NotImplementedError("Complete soft_assignments() TODOs.")


def hard_assignments(model: GaussianMixture, X_scaled: np.ndarray) -> np.ndarray:
    """
    Return hard cluster labels.

    TODO:
    - Use model.predict(X_scaled)
    """
    raise NotImplementedError("Complete hard_assignments() TODOs.")


def pca_2d(X_scaled: np.ndarray, random_state: int = 42):
    """
    Reduce scaled features to 2D for plotting.

    TODO:
    - Fit PCA(n_components=2, random_state=random_state) on X_scaled
    - Return (X_pca, pca_model)
    """
    raise NotImplementedError("Complete pca_2d() TODOs.")


def plot_clusters_2d(X_pca: np.ndarray, labels: np.ndarray, title: str = "GMM Clusters (PCA 2D)"):
    """
    Plot cluster assignments in PCA space.

    TODO:
    - Create a scatter plot of PC1 vs PC2
    - Color points by labels
    - Add title/labels
    """
    raise NotImplementedError("Complete plot_clusters_2d() TODOs.")


def plot_bic_curve(results_df: pd.DataFrame, title: str = "BIC by Number of Components"):
    """
    Plot BIC as a function of k.

    TODO:
    - Line plot with k on x-axis, bic on y-axis
    """
    raise NotImplementedError("Complete plot_bic_curve() TODOs.")


# ---------------------------------------------------------------------
# Step 3b: Interpret clusters via centroids (means)
# ---------------------------------------------------------------------

def cluster_centroids(model: GaussianMixture, scaler: StandardScaler, feature_names) -> pd.DataFrame:
    """
    Get cluster "centroids" (component means) in original feature units.

    Notes:
    - GMM means_ are in scaled space (if you fit on X_scaled).
    - Convert back using scaler.inverse_transform.

    TODO:
    - Inverse-transform model.means_ with scaler
    - Return a tidy DataFrame with:
        - cluster (int)
        - one column per feature
    """
    raise NotImplementedError("Complete cluster_centroids() TODOs.")

