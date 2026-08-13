"""Tests for the dense index wrapper.

Binary quantization packs 384 float dimensions into 48 bytes, so a shape or
dtype mistake surfaces as bad recall rather than an exception. These tests
assert the round trip actually retrieves what it should, at both quantizations.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicerag.index.dense import DenseIndex, binarize


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    rng = np.random.default_rng(0)
    v = rng.normal(size=(2000, 384)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestBinarize:
    def test_packs_to_one_bit_per_dimension(self, vectors):
        packed = binarize(vectors)
        assert packed.shape == (2000, 48)  # 384 bits / 8
        assert packed.dtype == np.uint8

    def test_preserves_sign_information(self):
        v = np.array([[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]], dtype=np.float32)
        assert np.unpackbits(binarize(v)).tolist() == [1, 0, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("quant", ["f32", "i8", "b1"])
class TestRoundTrip:
    def test_finds_itself(self, vectors, quant):
        """A vector's nearest neighbour must be itself at every quantization."""
        index = DenseIndex.build(vectors, quantization=quant, progress=False)
        keys, sims = index.search(vectors[7], top_k=5)

        assert len(index) == 2000
        assert keys[0] == 7, f"{quant} failed to self-retrieve"
        assert sims[0] > sims[-1], "similarities must be descending"

    def test_similarity_is_normalized(self, vectors, quant):
        index = DenseIndex.build(vectors, quantization=quant, progress=False)
        _, sims = index.search(vectors[0], top_k=10)
        assert np.all(sims <= 1.0001) and np.all(sims >= -1.0001)

    def test_save_and_view(self, vectors, quant, tmp_path):
        index = DenseIndex.build(vectors, quantization=quant, progress=False)
        path = tmp_path / f"idx_{quant}.usearch"
        index.save(path)

        loaded = DenseIndex.load(path, dim=384, quantization=quant, view=True)
        keys, _ = loaded.search(vectors[42], top_k=3)
        assert keys[0] == 42


@pytest.fixture(scope="module")
def clustered() -> np.ndarray:
    """Vectors with cluster structure, as real embeddings have.

    Uniform random vectors on a 384-d sphere are nearly all orthogonal, so their
    sign patterns carry almost no neighbourhood information and binary
    quantization looks far worse than it is in practice. Sentence embeddings sit
    in a small number of topical cones, which is what binary exploits.
    """
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(40, 384)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    members = np.repeat(centers, 50, axis=0)
    noisy = members + rng.normal(scale=0.35, size=members.shape).astype(np.float32)
    return noisy / np.linalg.norm(noisy, axis=1, keepdims=True)


def _recall_at_k(index, vectors: np.ndarray, truth: np.ndarray, k: int, step: int) -> float:
    hits = 0
    sampled = list(range(0, len(vectors), step))
    for i in sampled:
        keys, _ = index.search(vectors[i], top_k=k)
        hits += len(set(keys.tolist()) & set(truth[i].tolist()))
    return hits / (len(sampled) * k)


class TestQuantizationRecall:
    """Quantify each quantization's accuracy loss instead of assuming it.

    The hosting decision rests on this: binary is 8x smaller than int8, and
    these tests bound what that costs on embedding-like data. The authoritative
    number comes from scripts/eval_retrieval.py on the real corpus; this keeps
    a regression guard in the test suite.
    """

    @staticmethod
    def _truth(vectors: np.ndarray, k: int) -> np.ndarray:
        exact = vectors @ vectors.T
        np.fill_diagonal(exact, -np.inf)
        return np.argsort(-exact, axis=1)[:, :k]

    def test_int8_stays_close_to_exact(self, clustered):
        """int8 is the quantization we ship. Measured 84.8% recall@10 on the
        real corpus; this guards against a regression in the x127 scaling."""
        truth = self._truth(clustered, 10)
        index = DenseIndex.build(clustered, quantization="i8", progress=False)
        recall = _recall_at_k(index, clustered, truth, k=10, step=20)
        assert recall > 0.70, f"int8 recall@10 = {recall:.2%}"

    def test_int8_beats_raw_binary_decisively(self, clustered):
        """Records why this project does NOT ship binary quantization.

        Binary would be 8x smaller, which was tempting under a 512MB hosting
        budget. On the real corpus it scored 8.8% recall@10 against int8's
        84.8%, so the size win was never worth taking.
        """
        truth = self._truth(clustered, 10)
        i8 = DenseIndex.build(clustered, quantization="i8", progress=False)
        b1 = DenseIndex.build(clustered, quantization="b1", progress=False)

        i8_recall = _recall_at_k(i8, clustered, truth, k=10, step=20)
        b1_recall = _recall_at_k(b1, clustered, truth, k=10, step=20)
        assert i8_recall > b1_recall * 2, f"i8 {i8_recall:.2%} vs b1 {b1_recall:.2%}"


class TestAnisotropy:
    """E5 embeddings share a dominant direction, which is what breaks binary.

    Measured on the real corpus: mean cosine to the centroid is 0.867, so nearly
    every vector has the same sign pattern and Hamming distance becomes almost
    uninformative. Centering recovers some of it (8.8% -> 37.1%) but not enough
    to be usable. Kept as a test so the reasoning is executable, not folklore.
    """

    def test_centering_improves_binary_recall(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=(1500, 384)).astype(np.float32)
        # Inject a strong shared component, mimicking real embedding anisotropy.
        offset = np.abs(rng.normal(size=(1, 384))).astype(np.float32) * 2.0
        vectors = base * 0.3 + offset
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        centroid = vectors.mean(axis=0, keepdims=True)
        anisotropy = float(np.mean(vectors @ (centroid / np.linalg.norm(centroid)).T))
        assert anisotropy > 0.5, "fixture failed to reproduce anisotropy"

        exact = vectors @ vectors.T
        np.fill_diagonal(exact, -np.inf)
        truth = np.argsort(-exact, axis=1)[:, :10]

        centered = vectors - centroid
        centered /= np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9

        raw = DenseIndex.build(vectors, quantization="b1", progress=False)
        cen = DenseIndex.build(centered, quantization="b1", progress=False)

        raw_recall = _recall_at_k(raw, vectors, truth, k=10, step=15)
        cen_recall = _recall_at_k(cen, centered, truth, k=10, step=15)

        assert cen_recall > raw_recall, f"centering did not help: {raw_recall:.2%} -> {cen_recall:.2%}"
