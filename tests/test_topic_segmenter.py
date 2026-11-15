"""Tests for Hearst TextTiling Topic Segmenter (topic_segmenter.py)."""
from topic_segmenter import segment_transcript_topics


def test_segment_transcript_topics():
    # Construct a transcript with two distinct topics:
    # Topic 1: Artificial intelligence, machine learning, neural networks
    # Topic 2: Nutrition, carbohydrates, proteins, workout diet
    words = []
    t = 0.0

    # Topic 1 words
    ai_terms = ["artificial", "intelligence", "machine", "learning", "neural", "networks", "deep", "models", "data", "training"]
    for i in range(60):
        w = ai_terms[i % len(ai_terms)]
        words.append({"w": w, "s": round(t, 2), "e": round(t + 0.4, 2)})
        t += 0.5

    # Topic 2 words
    diet_terms = ["nutrition", "diet", "protein", "carbohydrates", "calories", "healthy", "food", "metabolism", "workout", "fitness"]
    for i in range(60):
        w = diet_terms[i % len(diet_terms)]
        words.append({"w": w, "s": round(t, 2), "e": round(t + 0.4, 2)})
        t += 0.5

    topics = segment_transcript_topics(words, block_size_words=15, min_topic_duration_sec=10.0)

    assert len(topics) >= 2
    assert "artificial" in topics[0].keywords or "intelligence" in topics[0].keywords
    assert "nutrition" in topics[-1].keywords or "protein" in topics[-1].keywords
    assert topics[0].start_time == 0.0
    assert topics[-1].end_time >= 50.0
