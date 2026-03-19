"""
Shared math utilities for Mess3 analysis.

KL rates between HMM processes, Procrustes alignment, subspace metrics.
"""

import numpy as np
from .hmm import Mess3HMM


# ---------------------------------------------------------------------------
# KL rates between Mess3 processes
# ---------------------------------------------------------------------------

def sequence_kl_rate(hmm1, hmm2):
    """
    Per-step KL rate h(P1 || P2) between two ergodic Mess3 processes.

    h = sum_i pi1[i] sum_x sum_j T1^x[i,j] * log(T1^x[i,j] / T2^x[i,j])

    This is the expected log-likelihood ratio per step under P1.
    After T tokens, P(wrong component | x_{1:T}) ~ exp(-T * h).
    """
    pi1 = hmm1.pi
    h = 0.0
    for x in range(3):
        T1 = hmm1.T_x[x]  # (3, 3)
        T2 = hmm2.T_x[x]  # (3, 3)
        for i in range(3):
            for j in range(3):
                if T1[i, j] > 1e-15:
                    t2_val = max(T2[i, j], 1e-15)
                    h += pi1[i] * T1[i, j] * np.log(T1[i, j] / t2_val)
    return h


def symmetric_kl_rate(hmm1, hmm2):
    """Symmetric KL rate: (h(P1||P2) + h(P2||P1)) / 2."""
    return (sequence_kl_rate(hmm1, hmm2) + sequence_kl_rate(hmm2, hmm1)) / 2


def emission_kl(hmm1, hmm2):
    """
    First-order KL approximation using marginal emission distributions.

    p(x) = sum_i pi[i] * E[i, x] where E[i,x] = P(emit x | state i).
    """
    p1 = hmm1.pi @ hmm1.E  # (3,)
    p2 = hmm2.pi @ hmm2.E  # (3,)
    kl = 0.0
    for x in range(3):
        if p1[x] > 1e-15:
            kl += p1[x] * np.log(p1[x] / max(p2[x], 1e-15))
    return kl


# ---------------------------------------------------------------------------
# Procrustes alignment
# ---------------------------------------------------------------------------

def procrustes_disparity(X, Y):
    """
    Procrustes disparity between two matched point clouds.

    Centers, scales to unit norm, finds optimal rotation via SVD.
    Returns disparity in [0, 1] — 0 means identical shapes.
    Uses scipy.spatial.procrustes under the hood.
    """
    from scipy.spatial import procrustes as scipy_procrustes
    _, _, disparity = scipy_procrustes(X, Y)
    return disparity


# ---------------------------------------------------------------------------
# Subspace separation metrics
# ---------------------------------------------------------------------------

def subspace_angles(acts_0, acts_1, n_components=3):
    """
    Principal angles between PCA subspaces of two activation sets.

    Returns angles in radians, sorted ascending. Small angles = overlapping subspaces.
    """
    from sklearn.decomposition import PCA

    pca0 = PCA(n_components=n_components).fit(acts_0)
    pca1 = PCA(n_components=n_components).fit(acts_1)

    # principal angles via SVD of the inner product of basis vectors
    V0 = pca0.components_  # (k, d)
    V1 = pca1.components_  # (k, d)
    M = V0 @ V1.T  # (k, k)
    svd_vals = np.linalg.svd(M, compute_uv=False)
    # clip for numerical safety
    svd_vals = np.clip(svd_vals, -1.0, 1.0)
    return np.arccos(svd_vals)


def cluster_separation(acts, labels):
    """
    Measure separation: silhouette score + linear classifier accuracy.

    acts: (N, d) activations
    labels: (N,) integer labels
    Returns dict with 'silhouette' and 'linear_acc'.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression

    sil = silhouette_score(acts, labels, sample_size=min(5000, len(acts)))

    # linear classifier with train/test split
    n = len(acts)
    idx = np.random.RandomState(0).permutation(n)
    split = int(0.8 * n)
    X_tr, X_te = acts[idx[:split]], acts[idx[split:]]
    y_tr, y_te = labels[idx[:split]], labels[idx[split:]]

    clf = LogisticRegression(max_iter=1000, random_state=0).fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)

    return {"silhouette": sil, "linear_acc": acc}


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def test_kl_properties():
    """Verify basic properties of the KL rate computation."""
    hmm_a = Mess3HMM(s=0.8, r=1.2)
    hmm_b = Mess3HMM(s=0.4, r=3.0)
    hmm_a_copy = Mess3HMM(s=0.8, r=1.2)

    # 1. KL with self = 0
    h_self = sequence_kl_rate(hmm_a, hmm_a_copy)
    assert abs(h_self) < 1e-10, f"KL(P||P) should be 0, got {h_self}"

    # 2. KL >= 0
    h_ab = sequence_kl_rate(hmm_a, hmm_b)
    h_ba = sequence_kl_rate(hmm_b, hmm_a)
    assert h_ab >= -1e-10, f"KL should be non-negative, got {h_ab}"
    assert h_ba >= -1e-10, f"KL should be non-negative, got {h_ba}"

    # 3. KL is asymmetric in general
    assert abs(h_ab - h_ba) > 0.001, f"KL should be asymmetric for different HMMs"

    # 4. Stationary distributions sum to 1
    assert abs(hmm_a.pi.sum() - 1.0) < 1e-10
    assert abs(hmm_b.pi.sum() - 1.0) < 1e-10

    # 5. Emission KL also non-negative and zero for identical
    assert abs(emission_kl(hmm_a, hmm_a_copy)) < 1e-10
    assert emission_kl(hmm_a, hmm_b) >= -1e-10

    # 6. Print actual values for sanity
    h_sym = symmetric_kl_rate(hmm_a, hmm_b)
    print(f"h(comp0 || comp1) = {h_ab:.6f}")
    print(f"h(comp1 || comp0) = {h_ba:.6f}")
    print(f"symmetric KL rate = {h_sym:.6f}")
    print(f"distinguishability horizon T* = 1/h = {1/h_sym:.1f} tokens")

    # all pairs
    hmm_c = Mess3HMM(s=0.6, r=0.3)
    hmms = [hmm_a, hmm_b, hmm_c]
    print("\npairwise symmetric KL rates:")
    for i in range(3):
        for j in range(i+1, 3):
            h = symmetric_kl_rate(hmms[i], hmms[j])
            print(f"  comp {i} vs {j}: h={h:.6f}, T*={1/h:.1f}")

    print("\nall tests passed.")


if __name__ == "__main__":
    test_kl_properties()
