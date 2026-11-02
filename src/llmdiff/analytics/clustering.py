"""Clustering of per-case score profiles, stdlib only.

K-means (Lloyd's algorithm with k-means++ seeding) and agglomerative
hierarchical clustering, plus silhouette scoring. Deterministic via seed.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from llmdiff.analytics.stats import mean

__all__ = [
    "euclidean",
    "kmeans",
    "kmeans_pp_seed",
    "agglomerative",
    "silhouette",
    "cluster_regressions",
]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError("vectors must have equal length")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def kmeans_pp_seed(points: Sequence[Sequence[float]], k: int, *, seed: int = 0) -> list[list[float]]:
    """k-means++ seeding: return k initial centroids."""
    n = len(points)
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n == 0:
        return []
    k = min(k, n)
    rng = random.Random(seed)
    centroids = [list(points[rng.randrange(n)])]
    while len(centroids) < k:
        dists = [min(euclidean(p, c) ** 2 for c in centroids) for p in points]
        total = sum(dists)
        if total == 0:
            centroids.append(list(points[rng.randrange(n)]))
            continue
        threshold = rng.random() * total
        acc = 0.0
        chosen = n - 1
        for i, d in enumerate(dists):
            acc += d
            if acc >= threshold:
                chosen = i
                break
        centroids.append(list(points[chosen]))
    return centroids


def kmeans(
    points: Sequence[Sequence[float]],
    k: int,
    *,
    max_iter: int = 100,
    seed: int = 0,
) -> tuple[list[int], list[list[float]]]:
    """Cluster ``points`` into ``k`` groups.

    Returns ``(labels, centroids)`` where labels[i] is the cluster index of
    points[i]. Deterministic for a given seed.
    """
    n = len(points)
    if n == 0:
        return [], []
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, n)
    centroids = kmeans_pp_seed(points, k, seed=seed)
    labels = [0] * n
    for _ in range(max_iter):
        changed = False
        for i, p in enumerate(points):
            best = min(range(k), key=lambda c: euclidean(p, centroids[c]))
            if labels[i] != best:
                labels[i] = best
                changed = True
        new_centroids: list[list[float]] = []
        for c in range(k):
            members = [points[i] for i in range(n) if labels[i] == c]
            if not members:
                new_centroids.append(centroids[c])
                continue
            dim = len(members[0])
            new_centroids.append([mean(m[j] for m in members) for j in range(dim)])
        centroids = new_centroids
        if not changed:
            break
    return labels, centroids


def agglomerative(
    points: Sequence[Sequence[float]],
    k: int,
    *,
    linkage: str = "average",
) -> list[int]:
    """Agglomerative hierarchical clustering cut at ``k`` clusters.

    linkage: ``"single"``, ``"complete"``, or ``"average"``.
    Returns labels in [0, k).
    """
    n = len(points)
    if n == 0:
        return []
    if k <= 0:
        raise ValueError("k must be >= 1")
    k = min(k, n)
    if linkage not in ("single", "complete", "average"):
        raise ValueError("linkage must be single/complete/average")
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = euclidean(points[i], points[j])
            dist[i][j] = dist[j][i] = d
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > k:
        best_pair = (0, 1)
        best_d = float("inf")
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                if linkage == "single":
                    d = min(dist[i][j] for i in clusters[a] for j in clusters[b])
                elif linkage == "complete":
                    d = max(dist[i][j] for i in clusters[a] for j in clusters[b])
                else:
                    d = sum(dist[i][j] for i in clusters[a] for j in clusters[b]) / (
                        len(clusters[a]) * len(clusters[b])
                    )
                if d < best_d:
                    best_d = d
                    best_pair = (a, b)
        a, b = best_pair
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = [0] * n
    for idx, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = idx
    return labels


def silhouette(points: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Mean silhouette coefficient in [-1, 1]; higher = better separated clusters."""
    n = len(points)
    if n == 0 or len(labels) != n:
        return 0.0
    clusters: dict[int, list[int]] = {}
    for i, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(i)
    if len(clusters) < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        own = clusters[labels[i]]
        if len(own) <= 1:
            s_i = 0.0
        else:
            a_i = sum(euclidean(points[i], points[j]) for j in own if j != i) / (len(own) - 1)
            b_i = min(
                sum(euclidean(points[i], points[j]) for j in members) / len(members)
                for lab, members in clusters.items()
                if lab != labels[i]
            )
            s_i = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0
        total += s_i
    return total / n


def cluster_regressions(
    score_rows: Sequence[Sequence[float]],
    case_ids: Sequence[str],
    *,
    k: int = 2,
    seed: int = 0,
) -> dict[int, list[str]]:
    """Cluster cases by their per-dimension score profile.

    Args:
        score_rows: One row of dimension scores per case.
        case_ids: Case id per row.
        k: Number of clusters.

    Returns:
        Mapping cluster index -> list of case ids in that cluster.
    """
    if len(score_rows) != len(case_ids):
        raise ValueError("score_rows and case_ids must have equal length")
    labels, _ = kmeans(score_rows, k, seed=seed)
    out: dict[int, list[str]] = {}
    for lab, cid in zip(labels, case_ids, strict=True):
        out.setdefault(lab, []).append(cid)
    return out
