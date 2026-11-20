"""Tests for Heuristic Viral Scoring & Hook Extraction (viral_brain.py)."""
from engine.intelligence.viral_brain import evaluate_viral_clip, generate_viral_hook, score_hook_strength


def test_score_hook_strength():
    high_hook = "Why did nobody tell you the 3 biggest mistakes in business?"
    score_high = score_hook_strength(high_hook)

    low_hook = "Today we are sitting down in the studio to talk about stuff."
    score_low = score_hook_strength(low_hook)

    assert score_high > score_low
    assert score_high >= 75


def test_generate_viral_hook():
    hook1 = generate_viral_hook("Why does everyone make this huge mistake in their code?")
    assert "mistake" in hook1.lower() or "why" in hook1.lower() or "stop" in hook1.lower()

    hook2 = generate_viral_hook("The secret trick that 10x my productivity.")
    assert "secret" in hook2.lower() or "trick" in hook2.lower() or "..." in hook2


def test_evaluate_viral_clip():
    text = (
        "Did you know that 90% of people make this catastrophic mistake when investing? "
        "They buy when the market is at its peak because of FOMO. "
        "In reality, the secret to massive wealth is doing the exact opposite. "
        "The lesson is simple: be fearful when others are greedy."
    )
    assessment = evaluate_viral_clip(text)

    assert assessment.overall_score >= 70
    assert assessment.hook_score >= 70
    assert len(assessment.viral_hook_text) > 0
    assert "#Shorts" in assessment.suggested_title or "Wait" in assessment.suggested_title
    assert "viral" in assessment.tiktok_description.lower()
