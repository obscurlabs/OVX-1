"""BM25 over a compact inverted index.

Why not rank_bm25: it retains the tokenized corpus as Python lists of strings.
At ~300k passages that is roughly 12M interned strings, comfortably over a
gigabyte of RAM, which does not fit the deployment budget and makes scoring slow
enough to threaten the latency target on its own.

This stores postings in flat numpy arrays (CSR-style) and, critically,
precomputes the full BM25 term-document weight at build time:

    w(t,d) = idf(t) * tf * (k1+1) / (tf + k1 * (1 - b + b * len(d)/avgdl))

Every factor there is known without the query. So retrieval degenerates to
"gather the postings of each query term and scatter-add their weights", which is
a couple of numpy operations rather than a Python loop over documents.

Tokenization covers Latin and Devanagari. Note that lexical matching is
deliberately monolingual - a Hindi query will not lexically match an English
passage. That is the dense index's job, and keeping the two honest about their
strengths is what makes fusing them worthwhile.
"""

from __future__ import annotations

import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

# Latin alphanumerics plus the Devanagari block (U+0900-U+097F).
_TOKEN = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")

# Very common terms carry near-zero idf but dominate posting-list length.
_STOPWORDS = frozenset(
    """a an the of in on at to for and or is are was were be been being it its this that
    these those with as by from what which who whom how why when where
    है हैं था थे थी का की के को में से पर और या एक यह वह जो कि तो ही भी हो होता होती
    """.split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


class BM25Index:
    """Inverted index with precomputed BM25 weights."""

    def __init__(
        self,
        vocab: dict[str, int],
        offsets: np.ndarray,
        doc_ids: np.ndarray,
        weights: np.ndarray,
        n_docs: int,
    ) -> None:
        self.vocab = vocab
        self.offsets = offsets  # int64[n_terms + 1], slice bounds into postings
        self.doc_ids = doc_ids  # int32[nnz]
        self.weights = weights  # float32[nnz], BM25 weight of term in doc
        self.n_docs = n_docs

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
        min_df: int = 1,
        progress_every: int = 50_000,
    ) -> BM25Index:
        n_docs = len(documents)
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_lengths = np.zeros(n_docs, dtype=np.float32)

        for doc_id, text in enumerate(documents):
            tokens = tokenize(text)
            doc_lengths[doc_id] = len(tokens)
            if not tokens:
                continue

            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, tf in counts.items():
                postings[term].append((doc_id, tf))

            if progress_every and (doc_id + 1) % progress_every == 0:
                print(f"    indexed {doc_id + 1:,}/{n_docs:,}", flush=True)

        avgdl = float(doc_lengths.mean()) if n_docs else 0.0
        # Guard against empty docs producing div-by-zero in the length norm.
        norm = k1 * (1.0 - b + b * (doc_lengths / max(avgdl, 1e-9)))

        terms = [t for t, plist in postings.items() if len(plist) >= min_df]
        terms.sort()

        vocab = {term: i for i, term in enumerate(terms)}
        nnz = sum(len(postings[t]) for t in terms)

        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        doc_id_arr = np.empty(nnz, dtype=np.int32)
        weight_arr = np.empty(nnz, dtype=np.float32)

        cursor = 0
        for i, term in enumerate(terms):
            plist = postings[term]
            df = len(plist)
            idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

            for doc_id, tf in plist:
                doc_id_arr[cursor] = doc_id
                weight_arr[cursor] = idf * (tf * (k1 + 1.0)) / (tf + norm[doc_id])
                cursor += 1
            offsets[i + 1] = cursor

        return cls(vocab, offsets, doc_id_arr, weight_arr, n_docs)

    # -- query --------------------------------------------------------------

    def search(self, query: str, top_k: int = 50) -> tuple[np.ndarray, np.ndarray]:
        """Return (doc_ids, scores) for the top_k matches, best first."""
        terms = tokenize(query)
        if not terms:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        # Gather every relevant posting first, then accumulate in ONE pass.
        #
        # The obvious implementation - np.add.at(scores, docs, weights) per term -
        # is a correctness-preserving performance trap: ufunc.at runs an unbuffered
        # Python-level loop and measured 3.9-5.6s on English queries, whose common
        # terms carry posting lists of 100k+ documents. np.bincount does the same
        # scatter-add in compiled code and is orders of magnitude faster.
        doc_slices: list[np.ndarray] = []
        weight_slices: list[np.ndarray] = []

        for term in terms:
            term_id = self.vocab.get(term)
            if term_id is None:
                continue
            start, end = self.offsets[term_id], self.offsets[term_id + 1]
            doc_slices.append(self.doc_ids[start:end])
            weight_slices.append(self.weights[start:end])

        if not doc_slices:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

        docs = np.concatenate(doc_slices)
        weights = np.concatenate(weight_slices)
        scores = np.bincount(docs, weights=weights, minlength=self.n_docs).astype(np.float32)

        # argpartition beats a full sort: we only need the top_k, not an ordering
        # of 300k documents, and this is on the latency-critical path.
        k = min(top_k, self.n_docs)
        candidates = np.argpartition(-scores, k - 1)[:k]
        candidates = candidates[scores[candidates] > 0]
        order = np.argsort(-scores[candidates])
        best = candidates[order]
        return best.astype(np.int32), scores[best]

    # -- persistence --------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "vocab": self.vocab,
                    "offsets": self.offsets,
                    "doc_ids": self.doc_ids,
                    "weights": self.weights,
                    "n_docs": self.n_docs,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        with path.open("rb") as fh:
            payload = pickle.load(fh)
        return cls(
            payload["vocab"],
            payload["offsets"],
            payload["doc_ids"],
            payload["weights"],
            payload["n_docs"],
        )

    @property
    def memory_mb(self) -> float:
        arrays = self.offsets.nbytes + self.doc_ids.nbytes + self.weights.nbytes
        # Rough: Python str objects dominate vocab cost.
        vocab_bytes = sum(len(t) + 49 for t in self.vocab) + len(self.vocab) * 8
        return (arrays + vocab_bytes) / 1024 / 1024
