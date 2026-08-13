"""Tests for the BM25 inverted index."""

from __future__ import annotations

import numpy as np
import pytest

from voicerag.index.lexical import BM25Index, tokenize

DOCS = [
    "The capital of India is New Delhi and it is very large",
    "Photosynthesis converts light energy into chemical energy in plants",
    "New Delhi hosts the Parliament of India",
    "The Pacific Ocean is the largest ocean on Earth",
    "Chlorophyll absorbs sunlight during photosynthesis in green plants",
]

HINDI_DOCS = [
    "भारत की राजधानी नई दिल्ली है",
    "प्रकाश संश्लेषण पौधों में होता है",
    "नई दिल्ली में संसद भवन स्थित है",
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build(DOCS, progress_every=0)


class TestTokenize:
    def test_splits_latin_and_lowercases(self):
        assert tokenize("The Capital, of INDIA!", drop_stopwords=False) == [
            "the", "capital", "of", "india",
        ]

    def test_handles_devanagari(self):
        tokens = tokenize("भारत की राजधानी", drop_stopwords=False)
        assert tokens == ["भारत", "की", "राजधानी"]

    def test_drops_stopwords_in_both_scripts(self):
        assert "the" not in tokenize("The capital")
        assert "है" not in tokenize("राजधानी है")

    def test_separates_mixed_script_runs(self):
        assert tokenize("Delhi दिल्ली 2024", drop_stopwords=False) == ["delhi", "दिल्ली", "2024"]

    def test_strips_punctuation_only_input(self):
        assert tokenize("!!! ... ???") == []


class TestSearch:
    def test_ranks_relevant_document_first(self, index):
        doc_ids, scores = index.search("photosynthesis chlorophyll", top_k=3)
        assert len(doc_ids) > 0
        assert doc_ids[0] in (1, 4)  # the two photosynthesis documents
        assert scores[0] > 0

    def test_scores_are_descending(self, index):
        _, scores = index.search("delhi india parliament", top_k=5)
        assert list(scores) == sorted(scores, reverse=True)

    def test_respects_top_k(self, index):
        doc_ids, _ = index.search("the of is in", top_k=2)
        assert len(doc_ids) <= 2

    def test_empty_query_returns_empty(self, index):
        doc_ids, scores = index.search("", top_k=5)
        assert len(doc_ids) == 0 and len(scores) == 0

    def test_stopword_only_query_returns_empty(self, index):
        doc_ids, _ = index.search("the of is", top_k=5)
        assert len(doc_ids) == 0

    def test_unknown_terms_return_empty(self, index):
        doc_ids, _ = index.search("zzzznonexistent quixotic", top_k=5)
        assert len(doc_ids) == 0

    def test_never_returns_zero_score_documents(self, index):
        _, scores = index.search("photosynthesis", top_k=5)
        assert all(s > 0 for s in scores)

    def test_multi_term_beats_single_term(self, index):
        """A document matching two query terms should outrank one matching one."""
        _, both = index.search("delhi parliament", top_k=1)
        _, one = index.search("parliament", top_k=1)
        assert both[0] > one[0]


class TestHindi:
    def test_retrieves_devanagari_documents(self):
        index = BM25Index.build(HINDI_DOCS, progress_every=0)
        doc_ids, scores = index.search("नई दिल्ली", top_k=3)
        assert len(doc_ids) > 0
        assert set(doc_ids[:2]) <= {0, 2}  # the two Delhi documents

    def test_cross_script_query_finds_nothing(self):
        """Documented limitation: lexical matching cannot bridge scripts.

        This is precisely the gap the dense index fills, and the reason fusing
        the two is worth the complexity rather than picking one.
        """
        index = BM25Index.build(HINDI_DOCS, progress_every=0)
        doc_ids, _ = index.search("new delhi capital", top_k=3)
        assert len(doc_ids) == 0


class TestPersistence:
    def test_roundtrip_preserves_results(self, index, tmp_path):
        path = tmp_path / "bm25.pkl"
        index.save(path)
        loaded = BM25Index.load(path)

        before_ids, before_scores = index.search("delhi india", top_k=3)
        after_ids, after_scores = loaded.search("delhi india", top_k=3)

        np.testing.assert_array_equal(before_ids, after_ids)
        np.testing.assert_allclose(before_scores, after_scores)

    def test_reports_memory(self, index):
        assert index.memory_mb > 0


class TestBuild:
    def test_handles_empty_documents(self):
        idx = BM25Index.build(["", "real content here", "   "], progress_every=0)
        doc_ids, _ = idx.search("content", top_k=3)
        assert list(doc_ids) == [1]

    def test_min_df_prunes_rare_terms(self):
        idx = BM25Index.build(DOCS, min_df=2, progress_every=0)
        # "chlorophyll" appears once, so it is pruned from the vocabulary.
        assert "chlorophyll" not in idx.vocab
        assert idx.search("chlorophyll", top_k=3)[0].size == 0

    def test_idf_penalises_ubiquitous_terms(self):
        """A term in every document should score lower than a selective one."""
        docs = ["common alpha", "common beta", "common gamma", "common alpha"]
        idx = BM25Index.build(docs, progress_every=0)
        _, common = idx.search("common", top_k=1)
        _, rare = idx.search("beta", top_k=1)
        assert rare[0] > common[0]
