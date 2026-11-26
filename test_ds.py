import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytest
import ds

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def songs_df():
    return ds.load_songs("data/taylor_album_songs.csv")


@pytest.fixture(scope="session")
def X(songs_df):
    X = ds.select_features(songs_df, features=ds.DEFAULT_FEATURES)
    return X


@pytest.fixture(scope="session")
def scaled(X):
    X_scaled, scaler = ds.scale_features(X)
    return X_scaled, scaler


# ---------------------------------------------------------------------
# Step 1: Load + Preprocess
# ---------------------------------------------------------------------

def test_load_songs_returns_dataframe(songs_df):
    assert isinstance(songs_df, pd.DataFrame)
    assert len(songs_df) > 0


def test_select_features_returns_expected_columns(X):
    assert isinstance(X, pd.DataFrame)
    assert len(X) > 0
    for col in ds.DEFAULT_FEATURES:
        assert col in X.columns, f"Missing required feature column: {col}"

    # features should be numeric after selection/coercion
    for col in ds.DEFAULT_FEATURES:
        assert pd.api.types.is_numeric_dtype(X[col]), f"{col} must be numeric"


def test_select_features_drops_missing(X):
    assert X.notna().all().all(), "Selected feature matrix must not contain NaNs"


def test_scale_features_outputs_numpy_and_scaler(scaled):
    X_scaled, scaler = scaled
    assert isinstance(X_scaled, np.ndarray)
    assert isinstance(scaler, StandardScaler)
    assert X_scaled.ndim == 2
    assert X_scaled.shape[1] == len(ds.DEFAULT_FEATURES)


def test_scaled_features_are_reasonably_standardized(scaled):
    X_scaled, _ = scaled
    means = X_scaled.mean(axis=0)
    stds = X_scaled.std(axis=0)

    # Loose tolerances to avoid brittleness
    assert np.all(np.abs(means) < 1e-1), "Scaled feature means should be near 0"
    assert np.all((stds > 0.7) & (stds < 1.3)), "Scaled feature std devs should be near 1"


# ---------------------------------------------------------------------
# Step 2: Fit GMM + Model Selection
# ---------------------------------------------------------------------

def test_fit_gmm_returns_fitted_model(scaled):
    X_scaled, _ = scaled
    model = ds.fit_gmm(X_scaled, n_components=3, random_state=42)
    assert isinstance(model, GaussianMixture)
    assert hasattr(model, "weights_"), "Fitted GMM should have learned weights_"
    assert model.means_.shape[0] == 3, "means_ should have one row per component"


def test_select_k_bic_returns_best_k_and_df(scaled):
    X_scaled, _ = scaled
    best_k, results_df = ds.select_k_bic(X_scaled, k_min=2, k_max=6, random_state=42)

    assert isinstance(best_k, int)
    assert 2 <= best_k <= 6

    assert isinstance(results_df, pd.DataFrame)
    assert set(["k", "bic"]).issubset(results_df.columns)
    assert len(results_df) == 5  # 2..6 inclusive
    assert results_df["bic"].notna().all()

    # best_k should correspond to minimum bic
    bic_min_k = int(results_df.loc[results_df["bic"].idxmin(), "k"])
    assert best_k == bic_min_k


def test_plot_bic_curve_runs(scaled):
    X_scaled, _ = scaled
    _, results_df = ds.select_k_bic(X_scaled, k_min=2, k_max=6, random_state=42)
    ds.plot_bic_curve(results_df, title="BIC Curve (Test)")
    plt.close("all")


# ---------------------------------------------------------------------
# Step 3: Soft assignments + PCA Visualization
# ---------------------------------------------------------------------

def test_soft_assignments_shape_and_sums(scaled):
    X_scaled, _ = scaled
    model = ds.fit_gmm(X_scaled, n_components=4, random_state=42)

    probs = ds.soft_assignments(model, X_scaled)

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (X_scaled.shape[0], 4)

    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Each row of probabilities must sum to 1"


def test_hard_assignments_shape_and_range(scaled):
    X_scaled, _ = scaled
    model = ds.fit_gmm(X_scaled, n_components=4, random_state=42)

    labels = ds.hard_assignments(model, X_scaled)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X_scaled.shape[0],)
    assert labels.min() >= 0 and labels.max() < 4, "Labels must be in [0, n_components-1]"


def test_pca_2d_outputs_correct_shape(scaled):
    X_scaled, _ = scaled
    X_pca, pca_model = ds.pca_2d(X_scaled, random_state=42)

    assert isinstance(X_pca, np.ndarray)
    assert X_pca.shape == (X_scaled.shape[0], 2)

    assert isinstance(pca_model, PCA)
    assert pca_model.n_components == 2


def test_plot_clusters_2d_runs(scaled):
    X_scaled, _ = scaled
    model = ds.fit_gmm(X_scaled, n_components=3, random_state=42)

    labels = ds.hard_assignments(model, X_scaled)
    X_pca, _ = ds.pca_2d(X_scaled, random_state=42)

    ds.plot_clusters_2d(X_pca, labels, title="Cluster Plot (Test)")
    plt.close("all")


# ---------------------------------------------------------------------
# Step 3b: Centroids / Interpretation
# ---------------------------------------------------------------------

def test_cluster_centroids_returns_tidy_df(scaled, X):
    X_scaled, scaler = scaled
    model = ds.fit_gmm(X_scaled, n_components=3, random_state=42)

    centroids = ds.cluster_centroids(model, scaler, feature_names=list(X.columns))

    assert isinstance(centroids, pd.DataFrame)
    assert "cluster" in centroids.columns
    assert len(centroids) == 3

    for col in X.columns:
        assert col in centroids.columns, f"Centroids missing feature column: {col}"
        assert pd.api.types.is_numeric_dtype(centroids[col]), f"{col} centroid must be numeric"

    # cluster labels should be 0..k-1 (order not strictly required, but typical)
    assert centroids["cluster"].nunique() == 3
    assert centroids["cluster"].min() >= 0
    assert centroids["cluster"].max() < 3
