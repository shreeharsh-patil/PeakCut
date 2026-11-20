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
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

QUESTION_WORDS = {"why", "how", "what", "who", "when", "where", "can", "is", "are", "do", "does", "did", "if", "kyu", "kyun", "kaise", "kya"}
HOOK_TRIGGERS = [
    r"\b(secret|mistake|never|always|nobody|everyone|stop|truth|hack|insane|crazy|worst|best)\b",
    r"\b(you won't believe|don't do this|did you know|this is why|the problem with|watch this)\b",
    r"\b\d+\s+(ways|things|secrets|reasons|mistakes|rules|steps|tips|tricks)\b",
    r"\b(10x|100%|million|billion|free|money|growth|clutch|ace|world record|breaking)\b",
    r"\b(raaz|galti|sach|asli|dhyan|mat karo)\b",
]
PAYOFF_TRIGGERS = [
    r"\b(because|result|turns out|finally|actually|here's why|in reality|the answer is)\b",
    r"\b(lesson|realized|discovered|solution|fix|takeaway|that's how|proved|won|beat|works)\b",
    r"\b(isliye|matlab|samajh|nateeja)\b",
]
SINGLE_WORD_FILLERS = {
    "um", "uh", "erm", "hmm", "like", "basically", "literally", "actually",
    "right", "yeah", "okay", "ok", "well", "honestly",
}
PHRASE_FILLERS = (
    "you know", "i mean", "sort of", "kind of", "you see", "at the end of the day",
)
FILLERS = SINGLE_WORD_FILLERS | set(PHRASE_FILLERS)

_DANGLING_START_RE = re.compile(
    r"^\s*(?:and|but|so|or|because|also|anyway|therefore|however|then|"
    r"that's why|thats why|he said|she said|they said|it was|this was)\b",
    re.IGNORECASE,
)
_DANGLING_END_WORDS = {
    "and", "but", "or", "so", "because", "if", "that", "which", "who",
    "the", "a", "an", "to", "of", "in", "on", "for", "with", "at", "by",
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
    text = (first_few_sentences or "").strip()
    if not text:
        return 20
    score = 45  # baseline
    lower = text.lower()
    first_tokens = re.findall(r"[a-zA-Z0-9']+", lower)
    first_word = first_tokens[0] if first_tokens else ""

    if "?" in text or first_word in QUESTION_WORDS:
        score += 20

    for pattern in HOOK_TRIGGERS:
        if re.search(pattern, lower):
            score += 15

    # Check for numerals / specific concrete quantities
    if re.search(r"\b\d+(?:[.,]\d+)?%?\b", lower):
        score += 10

    # Penalize weak/dangling openings that require prior context
    if _DANGLING_START_RE.match(text):
        score -= 15

    return max(0, min(100, score))


def score_payoff(text: str) -> int:
    """Evaluate whether the clip delivers a satisfying conclusion or insight."""
    text = (text or "").strip()
    if not text:
        return 20
    score = 50
    lower = text.lower()

    for pattern in PAYOFF_TRIGGERS:
        if re.search(pattern, lower):
            score += 18

    # Sentences ending cleanly with punctuation (including Hindi danda)
    if text.endswith((".", "!", "?", "।")):
        score += 12
    else:
        tokens = re.findall(r"[a-zA-Z']+", lower)
        if tokens and tokens[-1] in _DANGLING_END_WORDS:
            score -= 15

    return max(0, min(100, score))


def score_quotability(text: str) -> int:
    """Evaluate whether the clip contains punchy short sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?।]", text or "") if s.strip()]
    if not sentences:
        return 40

    short_punchy = sum(1 for s in sentences if 4 <= len(s.split()) <= 14)
    ratio = short_punchy / len(sentences)
    return min(100, int(40 + ratio * 60))


def score_cohesion(text: str) -> int:
    """Evaluate self-contained storytelling: clean opening, complete arc, clean ending."""
    cleaned = (text or "").strip()
    if not cleaned:
        return 20
    sentences = [s.strip() for s in re.split(r"[.!?।]+", cleaned) if s.strip()]
    score = 80 if len(sentences) >= 2 else 60
    if len(sentences) >= 3:
        score += 10
    if _DANGLING_START_RE.match(cleaned):
        score -= 20
    if cleaned.endswith((".", "!", "?", "।")):
        score += 10
    else:
        tokens = re.findall(r"[a-zA-Z']+", cleaned.lower())
        if tokens and tokens[-1] in _DANGLING_END_WORDS:
            score -= 20
    return max(0, min(100, score))


def score_density(text: str) -> int:
    """Ratio of meaningful content words vs filler speech (single & multi-word fillers)."""
    cleaned = (text or "").lower().strip()
    raw_words = re.findall(r"[^\s]+", cleaned)
    if not raw_words:
        return 20

    # Count multi-word phrase fillers first
    phrase_hits = 0
    for phrase in PHRASE_FILLERS:
        phrase_hits += len(re.findall(r"\b" + re.escape(phrase) + r"\b", cleaned))

    # Strip edge punctuation before checking single-word fillers
    single_hits = 0
    for w in raw_words:
        tok = re.sub(r"^[^\w]+|[^\w]+$", "", w)
        if tok in SINGLE_WORD_FILLERS:
            single_hits += 1

    # Each multi-word phrase filler accounts for 2 filler words
    filler_count = single_hits + phrase_hits * 2
    filler_ratio = min(1.0, filler_count / len(raw_words))

    # 0% fillers -> 100 density; 25% fillers -> 40 density
    density = max(20, int(100 - (filler_ratio * 240)))
    return min(100, density)


def generate_viral_hook(text: str) -> str:
    """Extract or formulate a high-CTR on-screen text hook."""
    cleaned = (text or "").strip()
    # Preserve sentence-ending punctuation so questions stay detectable!
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", cleaned) if s.strip()]
    first_sentence = sentences[0] if sentences else cleaned[:80]

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

    # Fallback to first 7 words (without trailing punctuation clutter)
    words = first_sentence.rstrip(".!?,;:").split()[:7]
    hook_str = " ".join(words) or "Watch This"
    return f"{hook_str}... ⚡"


def _strip_emojis_and_symbols(s: str) -> str:
    """Keep letters, combining marks (e.g. Hindi matras), digits, spaces, and basic punctuation."""
    kept = []
    for ch in s or "":
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N") or ch.isspace() or ch in ("'", "-"):
            kept.append(ch)
    return re.sub(r"\s+", " ", "".join(kept)).strip()


def generate_titles(text: str, hook: str) -> Tuple[str, str, str]:
    """Generate platform-specific titles and captions (supports Unicode/multilingual text)."""
    clean_hook = _strip_emojis_and_symbols(hook)
    if not clean_hook:
        clean_hook = _strip_emojis_and_symbols((text or "")[:60]) or "Highlight Moment"
    yt_title = f"{clean_hook} (Wait For It...) #Shorts"[:95]
    tiktok_desc = f"{clean_hook} 🤯 What do you think? Follow for more daily breakdowns! #viral #fyp #shorts"
    ig_desc = f"{clean_hook}\n\nDrop your thoughts below 👇\nFollow @openshorts for more daily insights.\n\n#reels #viral #shorts"
    return yt_title, tiktok_desc, ig_desc


def evaluate_viral_clip(transcript_text: str, duration: Optional[float] = None) -> ViralAssessment:
    """Evaluate transcript text and return complete virality metrics."""
    cleaned = (transcript_text or "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", cleaned) if s.strip()]
    first_part = " ".join(sentences[:2]) if len(sentences) >= 2 else cleaned[:140]

    hook = score_hook_strength(first_part)
    payoff = score_payoff(cleaned)
    quotability = score_quotability(cleaned)
    cohesion = score_cohesion(cleaned)
    density = score_density(cleaned)

    # Slight sweet-spot bonus when duration is in the ideal 20-45s Shorts window.
    dur_bonus = 0
    if duration is not None:
        try:
            d = float(duration)
            if 20.0 <= d <= 45.0:
                dur_bonus = 4
            elif 15.0 <= d <= 60.0:
                dur_bonus = 2
        except (TypeError, ValueError):
            pass

    overall = min(100, round(
        0.35 * hook
        + 0.25 * payoff
        + 0.20 * quotability
        + 0.10 * cohesion
        + 0.10 * density
        + dur_bonus
    ))

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
