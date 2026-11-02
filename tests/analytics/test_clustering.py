"""Tests for analytics.clustering — hand-computed expected values."""

from __future__ import annotations

import pytest

from llmdiff.analytics.clustering import (
    agglomerative,
    cluster_regressions,
    euclidean,
    kmeans,
    kmeans_pp_seed,
    silhouette,
)


def test_euclidean():
    assert euclidean([0, 0], [3, 4]) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        euclidean([1], [1, 2])


def test_kmeans_pp_seed_count():
    pts = [[0.0], [1.0], [10.0], [11.0]]
    seeds = kmeans_pp_seed(pts, 2, seed=1)
    assert len(seeds) == 2


def test_kmeans_two_blobs():
    pts = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [10.0, 10.0], [10.1, 10.0], [10.0, 10.1]]
    labels, _ = kmeans(pts, 2, seed=0)
    assert len(set(labels)) == 2
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_kmeans_deterministic():
    pts = [[float(i), float(i % 3)] for i in range(20)]
    l1, _ = kmeans(pts, 3, seed=7)
    l2, _ = kmeans(pts, 3, seed=7)
    assert l1 == l2


def test_kmeans_empty_and_bad_k():
    assert kmeans([], 2) == ([], [])
    with pytest.raises(ValueError):
        kmeans([[1.0]], 0)


def test_agglomerative():
    pts = [[0.0], [0.1], [5.0], [5.1]]
    labels = agglomerative(pts, 2)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_agglomerative_linkage_validation():
    with pytest.raises(ValueError):
        agglomerative([[0.0], [1.0]], 1, linkage="ward")


def test_silhouette_well_separated():
    pts = [[0.0], [0.1], [10.0], [10.1]]
    s = silhouette(pts, [0, 0, 1, 1])
    assert s > 0.9


def test_silhouette_single_cluster():
    assert silhouette([[0.0], [1.0]], [0, 0]) == 0.0


def test_cluster_regressions():
    rows = [[1.0, 1.0], [0.9, 1.0], [0.0, 0.0], [0.1, 0.0]]
    ids = ["a", "b", "c", "d"]
    groups = cluster_regressions(rows, ids, k=2, seed=0)
    assert sum(len(v) for v in groups.values()) == 4
    flat = {cid: g for g, cids in groups.items() for cid in cids}
    assert flat["a"] == flat["b"]
    assert flat["c"] == flat["d"]
    assert flat["a"] != flat["c"]


def test_cluster_regressions_length_mismatch():
    with pytest.raises(ValueError):
        cluster_regressions([[1.0]], ["a", "b"])
