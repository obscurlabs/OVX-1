"""Tests for the multi-strategy chunker.

The Hindi sentence-splitting tests are the important ones: a chunker that
silently fails to split Devanagari still "works" (it returns chunks), it just
returns useless whole-passage blobs. That failure is invisible without a test.
"""

from __future__ import annotations

import numpy as np
import pytest

from voicerag.index.chunking import (
    Chunk,
    ChunkConfig,
    ChunkStrategy,
    Granularity,
    SourcePassage,
    chunk_corpus,
    chunk_fixed_overlap,
    chunk_one,
    chunk_parent_grouped,
    chunk_semantic,
    chunk_sentence_window,
    split_sentences,
    strategy_stats,
    word_count,
)

HINDI = (
    "भारत की राजधानी नई दिल्ली है। यह देश का राजनीतिक केंद्र है। "
    "यहाँ संसद भवन स्थित है। दिल्ली की जनसंख्या लगभग दो करोड़ है।"
)
ENGLISH = (
    "The capital of India is New Delhi. It is the political centre of the country. "
    "Parliament House is located there. Delhi has a population of about 20 million."
)


class TestSentenceSplitting:
    def test_splits_hindi_on_danda(self):
        sents = split_sentences(HINDI)
        assert len(sents) == 4, f"danda split failed, got {len(sents)}: {sents}"
        assert sents[0].endswith("।")

    def test_splits_english_on_period(self):
        assert len(split_sentences(ENGLISH)) == 4

    def test_handles_mixed_script(self):
        mixed = "भारत की राजधानी नई दिल्ली है। The city is very large. यह बड़ा शहर है।"
        assert len(split_sentences(mixed)) == 3

    def test_merges_short_fragments(self):
        # "Dr." would otherwise become its own useless chunk.
        sents = split_sentences("Dr. Smith wrote the report. It was long enough to matter.")
        assert len(sents) == 2
        assert sents[0].startswith("Dr.")

    def test_empty_and_whitespace(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_single_sentence_without_terminator(self):
        assert split_sentences("no terminator here") == ["no terminator here"]

    def test_caps_runaway_unterminated_text(self):
        """Real corpus rows contain 1500-word blobs with no terminator.

        Without a cap the encoder truncates at 512 tokens and the tail is
        silently unretrievable, so the split must happen at index time.
        """
        blob = " ".join(f"w{i}" for i in range(500))  # zero terminators
        parts = split_sentences(blob, max_words=120)

        assert len(parts) == 5
        assert all(word_count(p) <= 120 for p in parts)
        # No text may be lost by the split.
        assert " ".join(parts).split() == blob.split()

    def test_cap_does_not_disturb_normal_text(self):
        assert split_sentences(ENGLISH, max_words=120) == split_sentences(ENGLISH)


class TestWordCount:
    def test_counts_devanagari_words(self):
        assert word_count("भारत की राजधानी") == 3

    def test_collapses_whitespace(self):
        assert word_count("a   b\n\nc") == 3


class TestSentenceWindow:
    def test_matches_narrow_answers_wide(self):
        p = SourcePassage(passage_id="p1", text=ENGLISH, lang="en")
        chunks = chunk_sentence_window(p, ChunkConfig())

        assert len(chunks) == 4
        middle = chunks[1]
        # Indexed text is one sentence; answer text pulls in the neighbours.
        assert middle.text == "It is the political centre of the country."
        assert "The capital of India" in middle.answer_text
        assert "Parliament House" in middle.answer_text
        assert middle.granularity is Granularity.SENTENCE

    def test_skips_single_sentence_passages(self):
        p = SourcePassage(passage_id="p1", text="Just one sentence here.", lang="en")
        assert chunk_sentence_window(p, ChunkConfig()) == []

    def test_window_clamps_at_boundaries(self):
        p = SourcePassage(passage_id="p1", text=ENGLISH, lang="en")
        chunks = chunk_sentence_window(p, ChunkConfig())
        # First chunk has no left neighbour, must not wrap around or crash.
        assert chunks[0].answer_text.startswith("The capital")


class TestFixedOverlap:
    def test_skips_short_passages(self):
        """The key anti-duplication guard: short passages get no window chunks."""
        p = SourcePassage(passage_id="p1", text=ENGLISH, lang="en")
        assert chunk_fixed_overlap(p, ChunkConfig()) == []

    def test_windows_long_passages_with_overlap(self):
        long_text = " ".join(f"word{i}" for i in range(300))
        p = SourcePassage(passage_id="p1", text=long_text, lang="en")
        cfg = ChunkConfig(long_passage_words=120, fixed_window_words=80, fixed_stride_words=56)
        chunks = chunk_fixed_overlap(p, cfg)

        assert len(chunks) > 1
        first, second = chunks[0].text.split(), chunks[1].text.split()
        overlap = set(first) & set(second)
        assert overlap, "stride must be smaller than window so chunks overlap"

    def test_no_chunk_below_min_words(self):
        long_text = " ".join(f"word{i}" for i in range(300))
        p = SourcePassage(passage_id="p1", text=long_text, lang="en")
        cfg = ChunkConfig(min_chunk_words=4)
        assert all(c.n_words >= cfg.min_chunk_words for c in chunk_fixed_overlap(p, cfg))


class TestSemantic:
    @staticmethod
    def _topical_embed(sentences):
        """Stub embedder: two clearly separated topics, no torch required."""
        out = []
        for s in sentences:
            out.append([1.0, 0.0] if "delhi" in s.lower() or "capital" in s.lower() else [0.0, 1.0])
        return np.array(out, dtype=np.float32)

    def test_splits_at_topic_shift(self):
        text = (
            "The capital of India is New Delhi. Delhi is a large city. "
            "Photosynthesis converts light energy. Chlorophyll absorbs sunlight."
        )
        p = SourcePassage(passage_id="p1", text=text, lang="en")
        chunks = chunk_semantic(p, ChunkConfig(semantic_min_sentences=4), self._topical_embed)

        assert len(chunks) >= 2
        assert all(c.strategy is ChunkStrategy.SEMANTIC for c in chunks)

    def test_skips_when_too_few_sentences(self):
        p = SourcePassage(passage_id="p1", text="One. Two.", lang="en")
        assert chunk_semantic(p, ChunkConfig(), self._topical_embed) == []

    def test_requires_embed_fn(self):
        p = SourcePassage(passage_id="p1", text=ENGLISH, lang="en")
        with pytest.raises(ValueError, match="embed_fn"):
            chunk_one(p, strategies=[ChunkStrategy.SEMANTIC])


class TestParentGrouped:
    def test_builds_document_from_query_siblings(self):
        passages = [
            SourcePassage(passage_id=f"p{i}", text=f"Passage {i} about Delhi.", lang="en", query_id="q1")
            for i in range(5)
        ]
        chunks = chunk_parent_grouped(passages, ChunkConfig())

        assert len(chunks) == 1
        doc = chunks[0]
        assert doc.granularity is Granularity.DOCUMENT
        assert doc.parent_id == "doc:q1"
        assert "Passage 0" in doc.text and "Passage 4" in doc.text

    def test_respects_word_budget(self):
        passages = [
            SourcePassage(passage_id=f"p{i}", text=" ".join(["word"] * 100), lang="en", query_id="q1")
            for i in range(10)
        ]
        chunks = chunk_parent_grouped(passages, ChunkConfig(), max_words=250)
        assert word_count(chunks[0].text) <= 250

    def test_ignores_passages_without_query_id(self):
        passages = [SourcePassage(passage_id="p1", text="text here", lang="en")]
        assert chunk_parent_grouped(passages, ChunkConfig()) == []


class TestCorpusOrchestration:
    def test_multi_granularity_index(self):
        passages = [
            SourcePassage(passage_id="p1", text=ENGLISH, lang="en", query_id="q1", query_type="description"),
            SourcePassage(passage_id="p2", text=HINDI, lang="hi", query_id="q1", query_type="description"),
        ]
        chunks = chunk_corpus(
            passages,
            strategies=[
                ChunkStrategy.PASSAGE,
                ChunkStrategy.SENTENCE_WINDOW,
                ChunkStrategy.PARENT_GROUPED,
            ],
        )

        produced = {c.strategy for c in chunks}
        assert produced == {
            ChunkStrategy.PASSAGE,
            ChunkStrategy.SENTENCE_WINDOW,
            ChunkStrategy.PARENT_GROUPED,
        }
        # Both scripts survive the pipeline.
        assert {c.lang for c in chunks} >= {"en", "hi"}

    def test_chunk_ids_are_deterministic(self):
        p = SourcePassage(passage_id="p1", text=ENGLISH, lang="en")
        assert [c.chunk_id for c in chunk_one(p)] == [c.chunk_id for c in chunk_one(p)]

    def test_chunk_ids_are_unique(self):
        passages = [
            SourcePassage(passage_id=f"p{i}", text=ENGLISH, lang="en", query_id="q1") for i in range(20)
        ]
        chunks = chunk_corpus(passages)
        assert len({c.chunk_id for c in chunks}) == len(chunks)

    def test_metadata_propagates(self):
        p = SourcePassage(
            passage_id="p1", text=ENGLISH, lang="en", query_id="q7", query_type="numeric", is_selected=True
        )
        for c in chunk_one(p):
            assert c.query_id == "q7"
            assert c.query_type == "numeric"
            assert c.is_selected is True

    def test_stats_report_per_strategy(self):
        passages = [SourcePassage(passage_id=f"p{i}", text=ENGLISH, lang="en") for i in range(5)]
        stats = strategy_stats(chunk_corpus(passages))

        assert ChunkStrategy.PASSAGE.value in stats
        assert stats[ChunkStrategy.PASSAGE.value]["count"] == 5
        assert stats[ChunkStrategy.PASSAGE.value]["mean_words"] > 0


class TestChunkModel:
    def test_answer_text_falls_back_to_text(self):
        c = Chunk(
            chunk_id="c1",
            text="body",
            passage_id="p1",
            lang="en",
            strategy=ChunkStrategy.PASSAGE,
            granularity=Granularity.PASSAGE,
        )
        assert c.answer_text == "body"
