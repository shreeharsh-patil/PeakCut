"""Phase 5 SEO Engine for ClipMind.

Generates high-CTR, algorithm-optimized metadata for YouTube Shorts:
1. Title generation following formula:
   [HOOK WORD] + [TOPIC] + [VALUE/CURIOSITY GAP] + [Emoji]
2. Description template with:
   - 2-line hook
   - Exact timestamps (00:00 Intro, 00:08 Main Point 1, 00:28 Main Point 2, 00:48 Shocking Reveal)
   - Call to action
   - Hashtags: #[Niche] #[Topic] #Shorts #[Trending1] #[Trending2]
3. Competitor tag extraction:
   - Top 15 tags from 3 highest-viewed competitors on topic + 5 long-tail keywords, totaling 450-500 chars.
4. Descriptive alt text for thumbnail variants (Variant A & Variant B).
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("clipmind.metadata_engine")

# Hook words strictly optimized for high CTR on YouTube Shorts
HOOK_WORDS = [
    "NEVER",
    "STOP",
    "WHY",
    "SECRET",
    "HOW",
    "WARNING",
    "THE TRUTH ABOUT",
    "UNBELIEVABLE",
    "DO NOT",
    "EXPOSED",
]

# Curiosity and value gap phrases creating urgent psychological tension
CURIOSITY_GAPS = [
    "Will Change Everything",
    "Nobody Tells You",
    "In 60 Seconds",
    "Before It's Too Late",
    "Blew My Mind",
    "Costs You Millions",
    "Explained Simply",
    "Shocked Everyone",
    "Is A Huge Mistake",
    "Will Blow Your Mind",
]

# High-converting emojis for YouTube Shorts
EMOJIS = ["🚨", "🤯", "⚠️", "💡", "🔥", "⚡", "😱", "📈", "🛑", "🤫"]

# Standard timestamps required for Shorts retention pacing
DEFAULT_TIMESTAMPS: Dict[str, str] = {
    "00:00": "Intro",
    "00:08": "Main Point 1",
    "00:28": "Main Point 2",
    "00:48": "Shocking Reveal",
}


@dataclass
class VideoMetadata:
    """Complete SEO metadata package for YouTube Shorts upload."""
    title: str
    title_options: List[str]
    description: str
    tags: List[str]
    tags_string: str
    hashtags: List[str]
    thumbnail_alt_a: str
    thumbnail_alt_b: str
    timestamps: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TIMESTAMPS))
    hook_word: str = ""
    curiosity_gap: str = ""
    niche: str = ""
    topic: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetadataEngine:
    """Phase 5 SEO & Metadata Engine."""

    def __init__(self):
        self.hook_words = list(HOOK_WORDS)
        self.curiosity_gaps = list(CURIOSITY_GAPS)
        self.emojis = list(EMOJIS)

    # -------------------------------------------------------------------------
    # 1. Title Generation: [HOOK WORD] + [TOPIC] + [VALUE/CURIOSITY GAP] + [Emoji]
    # -------------------------------------------------------------------------
    def generate_title(
        self,
        topic: str,
        hook_word: Optional[str] = None,
        curiosity_gap: Optional[str] = None,
        emoji: Optional[str] = None,
    ) -> str:
        """Generate title strictly using formula:

        [HOOK WORD] + [TOPIC] + [VALUE/CURIOSITY GAP] + [Emoji]
        """
        clean_topic = self._clean_topic_for_title(topic)
        hook = hook_word if hook_word else random.choice(self.hook_words)
        gap = curiosity_gap if curiosity_gap else random.choice(self.curiosity_gaps)
        em = emoji if emoji else random.choice(self.emojis)

        # Formula assembly
        title = f"{hook} {clean_topic} {gap} {em}"

        # Ensure title stays within YouTube Shorts mobile view optimal length (<= 85 chars)
        if len(title) > 85:
            # Shorten clean_topic if needed
            allowed_len = 85 - (len(hook) + len(gap) + len(em) + 4)
            if allowed_len > 15:
                clean_topic = clean_topic[:allowed_len].rstrip()
                title = f"{hook} {clean_topic} {gap} {em}"

        return title

    def generate_title_options(self, topic: str, niche: str = "", count: int = 5) -> List[str]:
        """Generate multiple distinct title candidates following the formula."""
        options: List[str] = []
        used_combinations = set()

        for i in range(count * 2):
            hook = self.hook_words[i % len(self.hook_words)]
            gap = self.curiosity_gaps[(i * 3) % len(self.curiosity_gaps)]
            em = self.emojis[(i * 2) % len(self.emojis)]
            combo = (hook, gap, em)
            if combo in used_combinations:
                continue
            used_combinations.add(combo)

            t = self.generate_title(topic, hook_word=hook, curiosity_gap=gap, emoji=em)
            if t not in options:
                options.append(t)
            if len(options) >= count:
                break

        return options

    # -------------------------------------------------------------------------
    # 2. Description Generation with 2-line hook, timestamps, CTA, and hashtags
    # -------------------------------------------------------------------------
    def generate_description(
        self,
        topic: str,
        niche: str,
        hook_lines: Optional[Tuple[str, str]] = None,
        timestamps: Optional[Dict[str, str]] = None,
        call_to_action: Optional[str] = None,
        trending_tags: Optional[List[str]] = None,
    ) -> str:
        """Generate structured Shorts description template.

        Sections:
        1. 2-line hook (high curiosity + stakes)
        2. Timestamps (00:00 Intro, 00:08 Main Point 1, 00:28 Main Point 2, 00:48 Shocking Reveal)
        3. Call to Action
        4. Hashtags: #[Niche] #[Topic] #Shorts #[Trending1] #[Trending2]
        """
        clean_topic = self._clean_topic_for_title(topic)
        clean_niche = niche.strip().title()

        # 2-line hook
        if hook_lines and len(hook_lines) >= 2:
            line1, line2 = hook_lines[0], hook_lines[1]
        else:
            line1 = f"Did you know the truth behind {clean_topic}? Most people get this completely wrong."
            line2 = f"Watch till the end to discover why {clean_niche} experts are stunned by this reveal!"

        # Timestamps
        ts_dict = timestamps or DEFAULT_TIMESTAMPS
        ts_lines = [f"{ts} {label}" for ts, label in ts_dict.items()]
        timestamps_block = "\n".join(ts_lines)

        # Call to Action
        cta = (
            call_to_action
            or "🔔 Subscribe to stay ahead of the curve! Like, comment your thoughts below, and share with a friend."
        )

        # Hashtags
        hashtags = self.generate_hashtags(topic=topic, niche=niche, trending_tags=trending_tags)
        hashtags_line = " ".join(hashtags)

        # Assemble full description template
        description = (
            f"{line1}\n"
            f"{line2}\n\n"
            f"{timestamps_block}\n\n"
            f"{cta}\n\n"
            f"{hashtags_line}"
        )
        return description

    def generate_hashtags(
        self,
        topic: str,
        niche: str,
        trending_tags: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate hashtags in format: #[Niche] #[Topic] #Shorts #[Trending1] #[Trending2]."""
        clean_niche_tag = "#" + re.sub(r"[^\w]", "", niche.title())
        clean_topic_words = "".join(re.findall(r"[A-Za-z0-9]+", topic.title()))
        clean_topic_tag = "#" + clean_topic_words[:25]

        # Trending tags fallback
        trend1 = "#Trending"
        trend2 = "#Viral"
        if trending_tags and len(trending_tags) >= 1:
            t1 = "#" + re.sub(r"[^\w]", "", trending_tags[0].title())
            if len(t1) > 2:
                trend1 = t1
        if trending_tags and len(trending_tags) >= 2:
            t2 = "#" + re.sub(r"[^\w]", "", trending_tags[1].title())
            if len(t2) > 2:
                trend2 = t2

        return [clean_niche_tag, clean_topic_tag, "#Shorts", trend1, trend2]

    # -------------------------------------------------------------------------
    # 3. Competitor Tag Mining (Top 15 + 5 long-tail, totaling 450-500 chars)
    # -------------------------------------------------------------------------
    def extract_competitor_tags(
        self,
        competitor_videos: Optional[List[Dict[str, Any]]],
        topic: str,
        niche: str,
    ) -> Dict[str, Any]:
        """Extract top 15 competitor tags from 3 highest-viewed competitors on topic

        plus 5 long-tail keywords, totaling 450-500 characters (max 500 chars).
        """
        # Step 1: Gather and rank tags from top 3 highest viewed competitor videos
        collected_tags: List[str] = []
        if competitor_videos:
            # Sort competitors by view count descending
            sorted_vids = sorted(
                competitor_videos,
                key=lambda v: int(v.get("views", 0)),
                reverse=True,
            )[:3]

            tag_counts: Dict[str, int] = {}
            for vid in sorted_vids:
                for tag in vid.get("tags", []):
                    clean_tag = tag.strip().lower()
                    if clean_tag and len(clean_tag) > 2:
                        tag_counts[clean_tag] = tag_counts.get(clean_tag, 0) + 1

            # Top 15 ranked by frequency
            ranked_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
            collected_tags = [t[0] for t in ranked_tags[:15]]

        # Ensure we have up to 15 competitor tags even if competitor data is sparse
        clean_topic_lower = topic.lower()
        clean_niche_lower = niche.lower()
        seed_competitor_pool = [
            clean_niche_lower,
            "shorts",
            "youtube shorts",
            clean_topic_lower,
            f"{clean_niche_lower} update",
            f"{clean_topic_lower} breakthrough",
            "viral video",
            f"{clean_niche_lower} facts",
            f"{clean_topic_lower} news",
            "trending now",
            f"new {clean_niche_lower}",
            f"{clean_topic_lower} 2026",
            "tech news",
            "must watch",
            "explained",
        ]
        for s_tag in seed_competitor_pool:
            if s_tag not in collected_tags and len(collected_tags) < 15:
                collected_tags.append(s_tag)

        # Step 2: Generate 5 high-converting long-tail keywords
        long_tail_keywords = [
            f"{topic.lower()} explained in 60 seconds",
            f"the truth about {topic.lower()} nobody tells you",
            f"how does {topic.lower()} actually work",
            f"best {niche.lower()} breakthrough secrets",
            f"why {topic.lower()} changes everything today",
        ]

        # Combine top 15 competitor tags + 5 long-tail keywords
        initial_tags = collected_tags[:15] + long_tail_keywords[:5]

        # Step 3: Calibrate tag string to total 450 - 500 characters
        # Pack or trim tags to stay in [450, 500] characters (YouTube max is 500)
        final_tags, tags_string = self._pack_tags_to_char_budget(
            initial_tags=initial_tags,
            topic=topic,
            niche=niche,
            min_chars=450,
            max_chars=500,
        )

        return {
            "tags_list": final_tags,
            "tags_string": tags_string,
            "char_count": len(tags_string),
            "tag_count": len(final_tags),
            "top_competitor_tag_count": min(15, len(collected_tags)),
            "long_tail_count": 5,
        }

    def _pack_tags_to_char_budget(
        self,
        initial_tags: List[str],
        topic: str,
        niche: str,
        min_chars: int = 450,
        max_chars: int = 500,
    ) -> Tuple[List[str], str]:
        """Pack tags so comma-separated string length is strictly between min_chars and max_chars (or <= max_chars)."""
        tags: List[str] = []
        seen = set()

        for t in initial_tags:
            cl = t.strip()
            if cl and cl.lower() not in seen:
                candidate_str = ", ".join(tags + [cl])
                if len(candidate_str) <= max_chars:
                    tags.append(cl)
                    seen.add(cl.lower())

        # If length is below min_chars, add complementary niche long-tail keywords
        filler_pool = [
            f"what you need to know about {topic.lower()}",
            f"{niche.lower()} explained step by step",
            f"shocking facts about {topic.lower()}",
            f"{niche.lower()} tutorial for beginners",
            f"why everyone is watching {topic.lower()}",
            f"top 10 {niche.lower()} secrets",
            f"{topic.lower()} full breakdown",
            f"viral {niche.lower()} reels",
            f"{topic.lower()} reaction",
            f"unbelievable {niche.lower()} update",
            f"{niche.lower()} tips",
            f"latest {topic.lower()}",
        ]

        for filler in filler_pool:
            if len(", ".join(tags)) >= min_chars:
                break
            if filler.lower() not in seen:
                candidate_str = ", ".join(tags + [filler])
                if len(candidate_str) <= max_chars:
                    tags.append(filler)
                    seen.add(filler.lower())

        # If still just under min_chars, pad with specific precision keywords
        extra_fillers = [
            f"{niche.lower()} trends",
            "viral shorts",
            f"{topic.lower()} deep dive",
            "knowledge",
            "learning",
            "insights",
        ]
        for extra in extra_fillers:
            candidate_str = ", ".join(tags + [extra])
            if len(candidate_str) <= max_chars:
                tags.append(extra)
            if len(", ".join(tags)) >= min_chars:
                break

        tags_string = ", ".join(tags)

        # Enforce hard ceiling of max_chars
        while len(tags_string) > max_chars and tags:
            tags.pop()
            tags_string = ", ".join(tags)

        return tags, tags_string

    # -------------------------------------------------------------------------
    # 4. Descriptive Alt Text for Thumbnails (Variant A and Variant B)
    # -------------------------------------------------------------------------
    def generate_thumbnail_alt_text(
        self,
        topic: str,
        variant: str = "A",
        hook_text: str = "",
        niche: str = "",
    ) -> str:
        """Generate descriptive alt text for thumbnails (Variant A and Variant B)."""
        clean_topic = self._clean_topic_for_title(topic)
        clean_niche = niche.title() if niche else "trending topic"
        display_hook = hook_text if hook_text else f"THE TRUTH ABOUT {clean_topic.upper()}"

        if variant.upper() == "A":
            # Variant A: High-Contrast Reaction & Curiosity Focus
            return (
                f"High-contrast YouTube Short thumbnail featuring a shocked facial expression, "
                f"bold neon yellow text reading '{display_hook}', vibrant glowing highlight arrows pointing to "
                f"a central visual element illustrating {clean_topic}, set against a dynamic {clean_niche.lower()} background."
            )
        else:
            # Variant B: Authority, Split-Comparison & Reveal Focus
            return (
                f"Dramatic YouTube Short thumbnail with bold red and black split-screen gradient, "
                f"large Montserrat bold white text reading '{display_hook}', glowing warning badge icons, "
                f"and side-by-side visual comparison exposing the truth about {clean_topic}."
            )

    # -------------------------------------------------------------------------
    # Master Metadata Generator
    # -------------------------------------------------------------------------
    def generate_metadata(
        self,
        topic: str,
        niche: str,
        competitor_videos: Optional[List[Dict[str, Any]]] = None,
        trending_tags: Optional[List[str]] = None,
        hook_word: Optional[str] = None,
        curiosity_gap: Optional[str] = None,
        emoji: Optional[str] = None,
    ) -> VideoMetadata:
        """Master method generating full SEO metadata package for YouTube Shorts."""
        selected_hook = hook_word if hook_word else random.choice(self.hook_words)
        selected_gap = curiosity_gap if curiosity_gap else random.choice(self.curiosity_gaps)
        selected_emoji = emoji if emoji else random.choice(self.emojis)

        # Primary title & 5 candidate options
        primary_title = self.generate_title(
            topic=topic,
            hook_word=selected_hook,
            curiosity_gap=selected_gap,
            emoji=selected_emoji,
        )
        title_options = self.generate_title_options(topic=topic, niche=niche, count=5)
        if primary_title not in title_options:
            title_options.insert(0, primary_title)

        # Description
        description = self.generate_description(
            topic=topic,
            niche=niche,
            trending_tags=trending_tags,
        )

        # Hashtags
        hashtags = self.generate_hashtags(
            topic=topic,
            niche=niche,
            trending_tags=trending_tags,
        )

        # Competitor tag extraction (450-500 chars)
        tag_data = self.extract_competitor_tags(
            competitor_videos=competitor_videos,
            topic=topic,
            niche=niche,
        )

        # Alt text for Variant A and B
        hook_for_thumb = f"{selected_hook} {topic.upper()}"
        alt_a = self.generate_thumbnail_alt_text(topic=topic, variant="A", hook_text=hook_for_thumb, niche=niche)
        alt_b = self.generate_thumbnail_alt_text(topic=topic, variant="B", hook_text=hook_for_thumb, niche=niche)

        return VideoMetadata(
            title=primary_title,
            title_options=title_options[:5],
            description=description,
            tags=tag_data["tags_list"],
            tags_string=tag_data["tags_string"],
            hashtags=hashtags,
            thumbnail_alt_a=alt_a,
            thumbnail_alt_b=alt_b,
            timestamps=dict(DEFAULT_TIMESTAMPS),
            hook_word=selected_hook,
            curiosity_gap=selected_gap,
            niche=niche,
            topic=topic,
        )

    # -------------------------------------------------------------------------
    # Internal Utility
    # -------------------------------------------------------------------------
    def _clean_topic_for_title(self, topic: str) -> str:
        """Strip brackets, emojis, or trailing punctuation from topic string."""
        clean = re.sub(r"\[.*?\]|\(.*?\)", "", topic)
        clean = re.sub(r"[^\w\s\-\']", "", clean)
        return " ".join(clean.split())
