# Ex-12: Clustering Taylor Swift Songs with Gaussian Mixture Models

>**Trigger warning:**<br>
>Discussion of Taylor Swift songs

## Objective

In this exercise, you will apply **Gaussian Mixture Models (GMMs)** to identify and interpret **musical clusters** within Taylor Swift’s discography based on Spotify audio features.

This exercise emphasizes:

* Understanding **soft clustering** and probabilistic cluster membership
* Using **model selection criteria** (BIC/AIC) to choose the number of clusters
* Visualizing high-dimensional data using **PCA**
* Interpreting clusters through **feature centroids**

By the end of this exercise, you should be able to explain *why* certain songs group together musically—not just how to run a clustering algorithm.

---

## Prerequisites

You should have:

* Python 3.9+
* GitHub access
* Familiarity with:

  * pandas and numpy
  * feature scaling
  * unsupervised learning concepts
  * basic plotting with matplotlib

Required libraries (installed via `requirements.txt`):

* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `pytest` (for autograding)

The dataset **`taylor_album_songs.csv`** is provided in the `data/` directory.

> ⚠️ You do **not** need to submit a Jupyter notebook for this exercise.
> All required work is completed in `ds.py`.

---

## Dataset

This exercise uses data from the **{taylor} R package** by W. Jake Thompson, which compiles Taylor Swift’s discography using:

* **Spotify Audio Features API**
* **Genius lyrics metadata**

We focus on **`taylor_album_songs`**, which includes:

* All songs from Taylor Swift’s **official studio albums**
* Excludes standalone singles and non-Taylor-owned versions
* Audio features standardized across albums

Dataset source:

* [https://taylor.wjakethompson.com/](https://taylor.wjakethompson.com/)
* TidyTuesday (2023-10-17)

---

## Audio Feature Metadata (Selected)

You will cluster songs using the following Spotify features:

| Feature            | Description                        |
| ------------------ | ---------------------------------- |
| `danceability`     | Suitability for dancing (0–1)      |
| `energy`           | Intensity and activity (0–1)       |
| `loudness`         | Average decibel level              |
| `speechiness`      | Presence of spoken words           |
| `acousticness`     | Likelihood the track is acoustic   |
| `instrumentalness` | Likelihood the track has no vocals |
| `liveness`         | Presence of an audience            |
| `valence`          | Musical positivity                 |
| `tempo`            | Beats per minute (BPM)             |

These features capture **style, mood, and production choices** rather than lyrical content.

---

## Research Question

> **How can Gaussian Mixture Models be used to identify distinct musical clusters in Taylor Swift’s songs, and what do these clusters reveal about stylistic variation across her catalog?**

More specifically:

* How many clusters best describe the data?
* How do clusters differ musically?
* What does *soft* clustering reveal that hard clustering does not?

---

## Step 1: Setup and Data Preprocessing

You will:

1. Load the dataset from `data/taylor_album_songs.csv`
2. Select the relevant numeric audio features
3. Handle missing values
4. **Standardize features** prior to clustering

> 💡 Why scaling matters:
> GMMs rely on distance and covariance—unscaled features can dominate the model.

---

## Step 2: Gaussian Mixture Modeling (GMM)

You will:

1. Fit GMMs with varying numbers of components
2. Use **Bayesian Information Criterion (BIC)** (or AIC) to select the optimal number of clusters
3. Fit a final GMM using the selected number of components

> 💡 GMMs differ from k-means by allowing:
>
> * Elliptical clusters
> * Probabilistic (soft) membership
> * Overlapping cluster boundaries

---

## Step 3: Analysis and Visualization

You will:

* Compute **soft assignments** (cluster probabilities)
* Compute **hard assignments** (most likely cluster)
* Reduce data to two dimensions using **PCA**
* Visualize clusters in 2D PCA space
* Extract and interpret **cluster centroids** in original feature units

---

## What You Must Implement

In `ds.py`, you are responsible for implementing:

1. **Data loading and feature selection**
2. **Feature scaling**
3. **GMM fitting**
4. **BIC-based model selection**
5. **Soft and hard cluster assignments**
6. **PCA-based visualization**
7. **Cluster centroid extraction and interpretation**

Your implementation must pass the provided **autograder tests** in `test_ds.py`.

---

## Discussion Prompts (Answer in D2L)

Reflect on the following:

* How do the identified clusters differ musically?
* How does BIC change as the number of clusters increases?
* What insights does **soft clustering** provide that hard clustering might miss?
* Do the clusters align with intuition about Taylor Swift’s musical eras or styles?

There is no single “correct” answer—focus on **clear reasoning supported by results**.

---

## Submission Instructions

* Submit **only** your completed `ds.py` file via GitHub
* Do **not** submit notebooks or additional scripts
* Ensure all tests pass before submitting

> The autograder evaluates **correctness**, **methodology**, and **code quality**.
