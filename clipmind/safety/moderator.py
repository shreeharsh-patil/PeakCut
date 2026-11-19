"""Phase 9 Safety & Compliance System for ClipMind.

Provides:
- OpenAI Moderation API checks with local heuristic fallbacks.
- Copyright & DMCA guardrails (fair use guidelines, music blacklist, trademark screening).
- Fallback Error Recovery Protocol with exponential backoff & jitter (3 retries).
- Fallback topic resolution when primary candidates fail safety checks.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from clipmind.core.config import get_config
from clipmind.core.key_rotator import get_active_key, record_usage, mark_rate_limited

logger = logging.getLogger("clipmind.safety")

# Local heuristic keywords for offline / fallback safety checks
UNSAFE_PATTERNS = [
    r"\b(suicide|self-harm|kill yourself|how to build a bomb|terrorist|hate speech)\b",
    r"\b(cp|pedophile|csam|doxxing|swatting|credit card dump)\b",
    r"\b(n-word|slur|genocide denial|torture)\b",
]

# Copyright and trademark sensitivity risk patterns
COPYRIGHT_RISK_PATTERNS = [
    r"\b(full movie|full episode|official soundtrack|leaked album|nintendo ost)\b",
    r"\b(cracked software|keygen|pirated|torrent download|warez)\b",
    r"\b(stream rip|taylor swift official audio|disney leak)\b",
]


@dataclass
class ModerationResult:
    """Detailed safety evaluation report."""
    is_safe: bool
    flagged_categories: List[str] = field(default_factory=list)
    category_scores: Dict[str, float] = field(default_factory=dict)
    copyright_risk: bool = False
    copyright_notes: List[str] = field(default_factory=list)
    recommendation: str = "PROCEED"  # PROCEED, RETRY_WITH_FALLBACK, REJECT
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "flagged_categories": self.flagged_categories,
            "category_scores": self.category_scores,
            "copyright_risk": self.copyright_risk,
            "copyright_notes": self.copyright_notes,
            "recommendation": self.recommendation,
            "reason": self.reason,
        }


class SafetyModerator:
    """Enforces content compliance, safety guidelines, and error recovery protocols."""

    def __init__(self):
        self.config = get_config()

    def check_text(self, text: str) -> ModerationResult:
        """Run text through OpenAI Moderation API or fallback heuristic engine."""
        if not text or not text.strip():
            return ModerationResult(is_safe=True, reason="Empty text passed")

        # 1. Run local copyright and trademark heuristics
        copyright_flagged = False
        copyright_notes = []
        for pattern in COPYRIGHT_RISK_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                copyright_flagged = True
                copyright_notes.append(f"Potential copyright/DMCA risk detected: '{match.group(0)}'")

        # 2. Check OpenAI Moderation API if key is available
        openai_key = get_active_key("openai")
        if openai_key and not openai_key.startswith("sim_"):
            try:
                url = "https://api.openai.com/v1/moderations"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openai_key}"
                }
                payload = {"input": text}
                resp = requests.post(url, headers=headers, json=payload, timeout=8)
                
                if resp.status_code == 200:
                    record_usage("openai", openai_key, units=1)
                    data = resp.json()
                    results = data.get("results", [{}])[0]
                    flagged = results.get("flagged", False)
                    categories = results.get("categories", {})
                    category_scores = results.get("category_scores", {})
                    
                    flagged_cats = [cat for cat, val in categories.items() if val]
                    
                    is_safe = (not flagged) and (not copyright_flagged)
                    rec = "PROCEED" if is_safe else "RETRY_WITH_FALLBACK"
                    reason = "Passed OpenAI Moderation" if is_safe else f"Flagged by moderation: {', '.join(flagged_cats or copyright_notes)}"
                    
                    return ModerationResult(
                        is_safe=is_safe,
                        flagged_categories=flagged_cats,
                        category_scores={k: round(v, 4) for k, v in category_scores.items()},
                        copyright_risk=copyright_flagged,
                        copyright_notes=copyright_notes,
                        recommendation=rec,
                        reason=reason,
                    )
                elif resp.status_code in (429, 403):
                    mark_rate_limited("openai", openai_key, reason=f"HTTP {resp.status_code} in moderation")
            except Exception as e:
                logger.warning(f"OpenAI Moderation API call failed, falling back to heuristic: {e}")

        # 3. Local heuristic fallback
        local_flagged_cats = []
        for pattern in UNSAFE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                local_flagged_cats.append(f"regex_violation: {match.group(0)}")

        is_safe = (len(local_flagged_cats) == 0) and (not copyright_flagged)
        rec = "PROCEED" if is_safe else "RETRY_WITH_FALLBACK"
        reason = "Passed local safety heuristic" if is_safe else f"Failed local safety check: {', '.join(local_flagged_cats + copyright_notes)}"

        return ModerationResult(
            is_safe=is_safe,
            flagged_categories=local_flagged_cats,
            copyright_risk=copyright_flagged,
            copyright_notes=copyright_notes,
            recommendation=rec,
            reason=reason,
        )

    def check_topic_and_script(self, topic: str, script: str = "") -> ModerationResult:
        """Evaluate topic and generated voiceover script combined."""
        combined_text = f"Title/Topic: {topic}\nScript: {script}"
        return self.check_text(combined_text)

    def select_safe_topic(
        self,
        candidate_topics: List[Dict[str, Any]] | List[str],
        fallback_topics: Optional[List[str]] = None
    ) -> Tuple[str, float, ModerationResult]:
        """Iterate through candidate topics, returning the first compliant topic.
        
        Falls back to curated fallback topics if all candidates fail.
        """
        fallbacks = fallback_topics or self.config.fallback_topics

        # Standardize candidate format: list of (topic_title, vps)
        items: List[Tuple[str, float]] = []
        for c in candidate_topics:
            if isinstance(c, dict):
                items.append((c.get("topic", c.get("title", "")), float(c.get("vps", c.get("score", 75.0)))))
            elif isinstance(c, str):
                items.append((c, 75.0))

        # Check candidate topics
        for topic_text, vps in items:
            result = self.check_text(topic_text)
            if result.is_safe:
                return topic_text, vps, result
            logger.warning(f"Candidate topic rejected by safety: '{topic_text}' ({result.reason})")

        # Candidates failed; try configured fallback topics
        logger.warning("All primary candidate topics failed safety checks. Utilizing fallback topics pool.")
        for fb in fallbacks:
            result = self.check_text(fb)
            if result.is_safe:
                return fb, 65.0, result

        # Ultimate clean default
        safe_default = "5 Mind-Blowing Scientific Facts You Never Knew"
        return safe_default, 60.0, ModerationResult(is_safe=True, reason="Safe default topic")

    def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None,
        fallback_action: Optional[Callable[..., Any]] = None,
        step_name: str = "PIPELINE_STEP",
        **kwargs: Any
    ) -> Any:
        """Phase 9 Fallback Error Recovery Protocol.
        
        Executes an action with up to max_retries attempts, exponential backoff,
        and random jitter. Calls fallback_action if all attempts fail.
        """
        retries = max_retries or self.config.scheduling.max_retries_per_cycle
        delay = initial_delay or self.config.scheduling.retry_delay_seconds

        last_exception: Optional[Exception] = None

        for attempt in range(1, retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                # Check for rate-limiting exceptions
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                    logger.warning(f"[{step_name}] Encountered rate limit on attempt {attempt}/{retries}: {e}")

                if attempt < retries:
                    # Exponential backoff with +/- 20% jitter
                    sleep_time = (delay * (2 ** (attempt - 1))) + random.uniform(0.5, 2.0)
                    logger.warning(
                        f"[{step_name}] Attempt {attempt} failed ({e}). Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    logger.error(f"[{step_name}] All {retries} retry attempts failed: {e}")

        # If fallback action is supplied, invoke it
        if fallback_action:
            logger.info(f"[{step_name}] Triggering fallback action recovery protocol...")
            return fallback_action(*args, **kwargs)

        if last_exception:
            raise last_exception
        raise RuntimeError(f"Step '{step_name}' failed after {retries} retries.")


_SAFETY_MODERATOR_INSTANCE: Optional[SafetyModerator] = None


def get_safety_moderator() -> SafetyModerator:
    """Retrieve singleton SafetyModerator instance."""
    global _SAFETY_MODERATOR_INSTANCE
    if _SAFETY_MODERATOR_INSTANCE is None:
        _SAFETY_MODERATOR_INSTANCE = SafetyModerator()
    return _SAFETY_MODERATOR_INSTANCE
