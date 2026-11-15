"""Marti Hearst TextTiling Conversational Topic Segmenter for OpenShorts.

Features:
- Identifies natural thematic topic boundaries and conversational shifts in long podcasts/interviews
- Divides word transcripts into pseudo-sentence token blocks
- Computes lexical cohesion and cosine similarity vectors across sliding blocks
- Calculates depth scores to find true topic boundaries (valley analysis)
- Ensures short clips don't cut in the middle of a continuous conceptual discussion
"""
from __future__ import annotations

import collections
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# Common stop words to exclude from lexical similarity
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his",
    "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your",
    "yours", "yourself", "yourselves", "yeah", "like", "know", "just", "um", "uh", "right", "okay",
}


@dataclass
class TopicSegment:
    topic_index: int
    start_time: float
    end_time: float
    duration: float
    keywords: List[str]
    sample_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _tokenize(word: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", word.lower())


def _cosine_similarity(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    dot = sum(v * vec2.get(k, 0) for k, v in vec1.items())
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def segment_transcript_topics(
    words: List[Dict[str, Any]],
    block_size_words: int = 25,
    min_topic_duration_sec: float = 30.0,
) -> List[TopicSegment]:
    """Segment a word transcript ({'w', 's', 'e'}) into conversational topic themes."""
    if not words:
        return []

    # Filter out stopwords and group into blocks
    token_words: List[Tuple[str, float, float]] = []
    for w in words:
        clean = _tokenize(w.get("w", ""))
        if clean and clean not in STOP_WORDS and len(clean) > 2:
            token_words.append((clean, float(w.get("s", 0.0)), float(w.get("e", 0.0))))

    if len(token_words) < block_size_words * 2:
        # Transcript too short for multi-topic segmentation
        start_t = words[0].get("s", 0.0)
        end_t = words[-1].get("e", 0.0)
        kw = [w[0] for w in token_words[:5]]
        sample = " ".join(w.get("w", "") for w in words[:20])
        return [
            TopicSegment(
                topic_index=1,
                start_time=start_t,
                end_time=end_t,
                duration=round(end_t - start_t, 1),
                keywords=kw,
                sample_text=sample,
            )
        ]

    # Create token blocks
    blocks: List[List[Tuple[str, float, float]]] = []
    for i in range(0, len(token_words), block_size_words):
        chunk = token_words[i : i + block_size_words]
        if len(chunk) >= 5:
            blocks.append(chunk)

    if len(blocks) < 2:
        start_t = words[0].get("s", 0.0)
        end_t = words[-1].get("e", 0.0)
        return [
            TopicSegment(
                topic_index=1,
                start_time=start_t,
                end_time=end_t,
                duration=round(end_t - start_t, 1),
                keywords=[w[0] for w in token_words[:5]],
                sample_text=" ".join(w.get("w", "") for w in words[:20]),
            )
        ]

    # Calculate frequency vectors
    vectors = [collections.Counter(w[0] for w in blk) for blk in blocks]

    # Calculate cohesion similarities between adjacent blocks
    similarities: List[float] = []
    for i in range(len(vectors) - 1):
        sim = _cosine_similarity(vectors[i], vectors[i + 1])
        similarities.append(sim)

    # Calculate depth scores at local valleys
    depths: List[float] = [0.0] * len(similarities)
    for i in range(len(similarities)):
        # Search left peak
        left_peak = similarities[i]
        for l in range(i - 1, -1, -1):
            if similarities[l] >= left_peak:
                left_peak = similarities[l]
            else:
                break
        # Search right peak
        right_peak = similarities[i]
        for r in range(i + 1, len(similarities)):
            if similarities[r] >= right_peak:
                right_peak = similarities[r]
            else:
                break
        depths[i] = (left_peak - similarities[i]) + (right_peak - similarities[i])

    avg_depth = sum(depths) / len(depths) if depths else 0.0
    threshold = avg_depth * 1.15

    # Identify cut boundaries
    cut_block_indices: List[int] = []
    for i, d in enumerate(depths):
        if d >= threshold:
            # Check local maximum depth
            is_max = True
            if i > 0 and depths[i - 1] > d:
                is_max = False
            if i < len(depths) - 1 and depths[i + 1] > d:
                is_max = False
            if is_max:
                cut_block_indices.append(i + 1)

    # Convert cut indices to TopicSegments
    segments: List[TopicSegment] = []
    start_block = 0
    topic_idx = 1

    for cut_idx in cut_block_indices + [len(blocks)]:
        if cut_idx <= start_block:
            continue

        seg_tokens = [tok for blk in blocks[start_block:cut_idx] for tok in blk]
        if not seg_tokens:
            continue

        start_time = seg_tokens[0][1]
        end_time = seg_tokens[-1][2]
        dur = end_time - start_time

        if dur >= min_topic_duration_sec or not segments:
            # Extract top 5 frequent keywords
            counts = collections.Counter(t[0] for t in seg_tokens)
            top_kw = [k for k, _ in counts.most_common(5)]
            sample = " ".join(t[0] for t in seg_tokens[:15])

            segments.append(
                TopicSegment(
                    topic_index=topic_idx,
                    start_time=round(start_time, 2),
                    end_time=round(end_time, 2),
                    duration=round(dur, 2),
                    keywords=top_kw,
                    sample_text=sample,
                )
            )
            topic_idx += 1
            start_block = cut_idx

    return segments
