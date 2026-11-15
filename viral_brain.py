"""Heuristic Viral Scoring, Hook Extraction & Title Generator for OpenShorts.

Features:
- 5-Axis Virality Rubric:
    1. Hook Strength (0-100): First 3s question, tension, bold claim, contradiction, numbers
    2. Emotional Payoff (0-100): Punchline, insight, revelation, or surprise
    3. Quotability (0-100): Punchy memorable one-liners
    4. Self-Contained Cohesion (0-100): Makes complete sense without external context
    5. Information Density (0-100): Content words vs filler words
- Algorithmic virality score calculation:
    score = round(0.35 * hook + 0.25 * payoff + 0.20 * quotability + 0.10 * cohesion + 0.10 * density)
- Algorithmic On-Screen Hook Generator (3-8 words with high-CTR emojis)
- SEO YouTube Shorts, TikTok, and Instagram Titles & Descriptions
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

QUESTION_WORDS = {"why", "how", "what", "who", "when", "where", "can", "is", "are", "do", "does", "did", "if"}
HOOK_TRIGGERS = [
    r"\b(secret|mistake|never|always|nobody|everyone|stop|truth|hack|insane|crazy|worst|best)\b",
    r"\b(you won't believe|don't do this|did you know|this is why|the problem with)\b",
    r"\b\d+ (ways|things|secrets|reasons|mistakes|rules)\b",
    r"\b(10x|100%|million|billion|free|money|growth)\b",
]
PAYOFF_TRIGGERS = [
    r"\b(because|result|turns out|finally|actually|here's why|in reality|the answer is)\b",
    r"\b(lesson|realized|discovered|solution|fix)\b",
]
FILLERS = {
    "um", "uh", "like", "you know", "i mean", "sort of", "kind of", "basically",
    "literally", "actually", "right", "yeah", "okay",
}


@dataclass
class ViralAssessment:
    overall_score: int
    hook_score: int
    payoff_score: int
    quotability_score: int
    cohesion_score: int
    density_score: int
    viral_hook_text: str
    suggested_title: str
    tiktok_description: str
    instagram_description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def score_hook_strength(first_few_sentences: str) -> int:
    """Evaluate hook strength based on curiosity, questions, tension, and trigger words."""
    score = 45  # baseline
    lower = first_few_sentences.lower()

    if "?" in first_few_sentences or any(lower.startswith(q) for q in QUESTION_WORDS):
        score += 20

    for pattern in HOOK_TRIGGERS:
        if re.search(pattern, lower):
            score += 15

    # Check for numerals
    if re.search(r"\b\d+\b", lower):
        score += 10

    return min(100, score)


def score_payoff(text: str) -> int:
    """Evaluate whether the clip delivers a satisfying conclusion or insight."""
    score = 50
    lower = text.lower()

    for pattern in PAYOFF_TRIGGERS:
        if re.search(pattern, lower):
            score += 18

    # Sentences ending cleanly with punctuation
    if text.rstrip().endswith((".", "!", "?")):
        score += 12

    return min(100, score)


def score_quotability(text: str) -> int:
    """Evaluate whether the clip contains punchy short sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    if not sentences:
        return 40

    short_punchy = sum(1 for s in sentences if 4 <= len(s.split()) <= 12)
    ratio = short_punchy / len(sentences)
    return min(100, int(40 + ratio * 60))


def score_density(text: str) -> int:
    """Ratio of meaningful content words vs filler speech."""
    words = text.lower().split()
    if not words:
        return 50

    filler_count = sum(1 for w in words if w in FILLERS)
    filler_ratio = filler_count / len(words)

    # 0% fillers -> 100 density; 25% fillers -> 40 density
    density = max(20, int(100 - (filler_ratio * 240)))
    return min(100, density)


def generate_viral_hook(text: str) -> str:
    """Extract or formulate a high-CTR on-screen text hook."""
    sentences = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    first_sentence = sentences[0] if sentences else text[:80]

    # If first sentence is a punchy question
    if "?" in first_sentence and len(first_sentence.split()) <= 10:
        return f"{first_sentence.rstrip('?')}? 🤯"

    # Pattern based hook
    lower = first_sentence.lower()
    if "mistake" in lower or "don't" in lower or "stop" in lower:
        return "Stop Doing This! 🚫"
    if "secret" in lower or "nobody" in lower:
        return "Nobody Tells You This... 🤫"
    if "how" in lower or "why" in lower:
        return "The Real Reason Why 😳"

    # Fallback to first 6 words
    words = first_sentence.split()[:7]
    hook_str = " ".join(words)
    return f"{hook_str}... ⚡"


def generate_titles(text: str, hook: str) -> Tuple[str, str, str]:
    """Generate platform-specific titles and captions."""
    clean_hook = re.sub(r"[^a-zA-Z0-9\s]", "", hook).strip()
    yt_title = f"{clean_hook} (Wait For It...) #Shorts"[:95]
    tiktok_desc = f"{clean_hook} 🤯 What do you think? Follow for more daily breakdowns! #viral #fyp #shorts"
    ig_desc = f"{clean_hook}\n\nDrop your thoughts below 👇\nFollow @openshorts for more daily insights.\n\n#reels #viral #shorts"
    return yt_title, tiktok_desc, ig_desc


def evaluate_viral_clip(transcript_text: str) -> ViralAssessment:
    """Evaluate transcript text and return complete virality metrics."""
    sentences = [s.strip() for s in re.split(r"[.!?]", transcript_text) if s.strip()]
    first_part = " ".join(sentences[:2]) if len(sentences) >= 2 else transcript_text[:120]

    hook = score_hook_strength(first_part)
    payoff = score_payoff(transcript_text)
    quotability = score_quotability(transcript_text)
    cohesion = 85 if len(sentences) >= 2 else 65
    density = score_density(transcript_text)

    overall = round(
        0.35 * hook
        + 0.25 * payoff
        + 0.20 * quotability
        + 0.10 * cohesion
        + 0.10 * density
    )

    hook_text = generate_viral_hook(transcript_text)
    yt_title, tiktok_desc, ig_desc = generate_titles(transcript_text, hook_text)

    return ViralAssessment(
        overall_score=overall,
        hook_score=hook,
        payoff_score=payoff,
        quotability_score=quotability,
        cohesion_score=cohesion,
        density_score=density,
        viral_hook_text=hook_text,
        suggested_title=yt_title,
        tiktok_description=tiktok_desc,
        instagram_description=ig_desc,
    )
