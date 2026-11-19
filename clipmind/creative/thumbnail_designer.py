"""Phase 4 CTR-Maximized Thumbnail Generation Engine for ClipMind.

Implements the high-CTR design formula:
1. Bold reaction element (stickers, emojis, warning badges with emotional contrast).
2. 3-5 word punchy text overlay (using Anton / Impact / Montserrat bold typography).
3. High-contrast neon/yellow/red styling with heavy black outlines (12-18px) and Gaussian drop shadows.
4. Emotion arrow / indicator pointing directly to the focal element.
5. Watermark / branding overlay (using assets/watermark.png or dynamic badge).
6. Pillow-based rendering engine + AI background art generator prompt (DALL-E 3 / Midjourney / Gemini).
7. A/B Testing Variant Generator:
   - Variant A: Curiosity / Shock Hook, Electric Neon Yellow + Crimson, Left text / Right focal point, Red arrow.
   - Variant B: Urgency / Secret Hook, Acid Green / Neon Cyan + Hot Magenta, Centered / Top text, Target reticle.
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    Image = ImageChops = ImageDraw = ImageEnhance = ImageFilter = ImageFont = None  # type: ignore
    PIL_AVAILABLE = False

# Core integrations
try:
    from clipmind.core.config import Config, get_config
except ImportError:
    class Config:  # type: ignore
        def __init__(self):
            self.dry_run = False
            self.paths = None

    def get_config():  # type: ignore
        return Config()

try:
    from clipmind.core.key_rotator import get_active_key, record_usage, mark_rate_limited
except ImportError:
    def get_active_key(service: str) -> Optional[str]:
        return os.environ.get(f"{service.upper()}_API_KEY")

    def record_usage(service: str, key: str, units: int = 1) -> None:
        pass

    def mark_rate_limited(service: str, key: str, duration_seconds: float = 3600.0, reason: str = "") -> None:
        pass

logger = logging.getLogger("clipmind.creative.thumbnail")

# Default Dimensions
THUMB_WIDTH = 1280
THUMB_HEIGHT = 720
MAX_FILE_BYTES = 2 * 1024 * 1024  # YouTube 2MB upload limit


@dataclass
class ThumbnailVariant:
    """Metadata and paths for a generated thumbnail variant."""
    variant_id: str  # "A" or "B"
    image_path: Path
    width: int = THUMB_WIDTH
    height: int = THUMB_HEIGHT
    headline_text: str = ""
    reaction_label: str = ""
    color_palette: Dict[str, str] = field(default_factory=dict)
    focal_point: str = "right"  # "left", "right", "center"
    arrow_target: Optional[Tuple[int, int]] = None
    ai_prompt: str = ""
    ctr_prediction_score: float = 0.125
    design_hypothesis: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "image_path": str(self.image_path),
            "width": self.width,
            "height": self.height,
            "headline_text": self.headline_text,
            "reaction_label": self.reaction_label,
            "color_palette": self.color_palette,
            "focal_point": self.focal_point,
            "arrow_target": list(self.arrow_target) if self.arrow_target else None,
            "ai_prompt": self.ai_prompt,
            "ctr_prediction_score": round(self.ctr_prediction_score, 4),
            "design_hypothesis": self.design_hypothesis,
            "created_at": self.created_at,
        }


@dataclass
class ThumbnailResult:
    """Pair of A/B test thumbnail variants generated for a video cycle."""
    variant_a: ThumbnailVariant
    variant_b: ThumbnailVariant
    topic: str
    output_dir: Path
    all_variants: List[ThumbnailVariant] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "output_dir": str(self.output_dir),
            "variant_a": self.variant_a.to_dict(),
            "variant_b": self.variant_b.to_dict(),
            "all_variants": [v.to_dict() for v in self.all_variants],
        }


def is_serverless() -> bool:
    """Detect if executing inside Vercel or AWS Lambda serverless function."""
    return bool(
        os.environ.get("VERCEL") == "1"
        or os.environ.get("VERCEL_ENV")
        or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        or os.environ.get("LAMBDA_TASK_ROOT")
    )


class ThumbnailDesigner:
    """Engine for generating CTR-maximized A/B thumbnail variants."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        if is_serverless():
            tmp_root = Path(tempfile.gettempdir()) / "clipmind"
            self.output_dir = tmp_root / "output" / "thumbnails"
        elif hasattr(self.config, "paths") and self.config.paths:
            self.output_dir = Path(self.config.paths.output_dir) / "thumbnails"
        else:
            self.output_dir = Path("output/thumbnails")

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            tmp_root = Path(tempfile.gettempdir()) / "clipmind"
            self.output_dir = tmp_root / "output" / "thumbnails"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Asset Paths (aware of project root and serverless /var/task root)
        project_root = Path(__file__).resolve().parent.parent.parent
        self.font_path = self._locate_font(project_root)
        self.watermark_path = self._locate_watermark(project_root)

    def _locate_font(self, project_root: Optional[Path] = None) -> Path:
        """Locate Anton-Regular or fallback font."""
        root = project_root or Path(__file__).resolve().parent.parent.parent
        candidates = [
            root / "fonts" / "Anton-Regular.ttf",
            Path("fonts/Anton-Regular.ttf"),
            Path("/var/task/fonts/Anton-Regular.ttf"),
            Path("../fonts/Anton-Regular.ttf"),
            Path("C:/Windows/Fonts/impact.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return Path("fonts/Anton-Regular.ttf")

    def _locate_watermark(self, project_root: Optional[Path] = None) -> Path:
        """Locate watermark image asset across local and serverless roots."""
        root = project_root or Path(__file__).resolve().parent.parent.parent
        candidates = [
            root / "assets" / "watermark.png",
            Path("assets/watermark.png"),
            Path("/var/task/assets/watermark.png"),
        ]
        for c in candidates:
            if c.exists():
                return c
        return Path("assets/watermark.png")

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load truetype font or fallback to default."""
        try:
            if self.font_path.exists():
                return ImageFont.truetype(str(self.font_path), size)
            # Try Windows system Impact
            if os.name == "nt" and os.path.exists("C:/Windows/Fonts/impact.ttf"):
                return ImageFont.truetype("C:/Windows/Fonts/impact.ttf", size)
        except Exception as e:
            logger.debug(f"Could not load custom truetype font: {e}")
        return ImageFont.load_default()

    # -----------------------------------------------------------------------
    # AI Prompt Synthesis
    # -----------------------------------------------------------------------

    def generate_ai_prompt(self, topic: str, variant_id: str = "A") -> str:
        """Generate high-performing AI image generation prompt for backgrounds."""
        clean_topic = topic.strip().rstrip(".!?")
        if variant_id.upper() == "A":
            return (
                f"Cinematic ultra-realistic 8k YouTube thumbnail background about {clean_topic}. "
                "Dramatic split lighting with intense electric yellow and deep midnight obsidian contrast. "
                "The right side features a striking glowing focal object representing the discovery with volumetric rim lighting. "
                "The left 45% of the frame is clean, softly blurred, dark negative space for high-impact text overlay. "
                "Octane render, photorealistic, sharp subject separation, extreme depth of field, 16:9 ratio. "
                "ABSOLUTELY NO TEXT, NO LETTERS, NO TYPOGRAPHY, NO LOGOS."
            )
        else:
            return (
                f"Hyper-intense dramatic YouTube thumbnail scene revealing the secret of {clean_topic}. "
                "Neon acid green and electric magenta color grade with cinematic lens flare and subtle smoke atmosphere. "
                "The bottom-right shows an incredible glowing core phenomenon under investigation. "
                "The top and center have clean dark negative space with soft atmospheric fog for bold title placement. "
                "Cinematic 3D render, Unreal Engine 5, hyper-detailed, high visual tension, 16:9 ratio. "
                "ABSOLUTELY NO TEXT, NO LETTERS, NO WATERMARKS."
            )

    # -----------------------------------------------------------------------
    # Background Generation (AI or Procedural)
    # -----------------------------------------------------------------------

    def generate_background_image(
        self,
        topic: str,
        variant_id: str,
        width: int = THUMB_WIDTH,
        height: int = THUMB_HEIGHT,
    ) -> Image.Image:
        """Fetch background via OpenAI DALL-E 3 if key available, or render procedural high-contrast canvas."""
        openai_key = get_active_key("openai")
        if openai_key and not openai_key.startswith("sim_") and not getattr(self.config, "dry_run", False):
            try:
                import requests
                prompt = self.generate_ai_prompt(topic, variant_id)
                headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
                payload = {
                    "model": "dall-e-3",
                    "prompt": prompt,
                    "n": 1,
                    "size": "1792x1024",  # Closest to 16:9
                    "quality": "standard"
                }
                resp = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=payload, timeout=25)
                if resp.status_code == 200:
                    record_usage("openai", openai_key, units=1000)
                    data = resp.json()
                    img_url = data["data"][0]["url"]
                    img_resp = requests.get(img_url, timeout=15)
                    if img_resp.status_code == 200:
                        downloaded = Image.open(io.BytesIO(img_resp.content)).convert("RGBA")
                        logger.info(f"Generated AI DALL-E 3 background for Variant {variant_id}")
                        return downloaded.resize((width, height), Image.LANCZOS)
                elif resp.status_code in (429, 403):
                    mark_rate_limited("openai", openai_key, reason=f"HTTP {resp.status_code}")
            except Exception as e:
                logger.warning(f"AI background image generation failed: {e}, falling back to procedural engine")

        # Procedural Canvas
        return self._render_procedural_background(variant_id, width, height)

    def _render_procedural_background(self, variant_id: str, width: int, height: int) -> Image.Image:
        """Procedurally render a rich, high-contrast cinematic canvas using Pillow."""
        bg = Image.new("RGBA", (width, height), (12, 14, 20, 255))
        draw = ImageDraw.Draw(bg)

        if variant_id.upper() == "A":
            # Deep Obsidian to Navy Gradient
            for y in range(height):
                ratio = y / height
                r = int(8 + 10 * ratio)
                g = int(10 + 15 * ratio)
                b = int(22 + 35 * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

            # Electric Yellow / Amber Radial Glow on Right Focal Area
            focal_x = int(width * 0.75)
            focal_y = int(height * 0.50)
            glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)

            for rad in range(350, 0, -15):
                alpha = int(45 * (1.0 - (rad / 350.0)))
                glow_draw.ellipse(
                    [focal_x - rad, focal_y - rad, focal_x + rad, focal_y + rad],
                    fill=(255, 200, 0, alpha)
                )
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(30))
            bg = Image.alpha_composite(bg, glow_layer)

            # Dramatic Rim Light Arc
            draw = ImageDraw.Draw(bg)
            draw.arc([focal_x - 180, focal_y - 180, focal_x + 180, focal_y + 180], start=-60, end=120, fill=(255, 240, 100, 220), width=6)

        else:
            # Dark Charcoal to Deep Emerald/Teal Gradient
            for y in range(height):
                ratio = y / height
                r = int(14 + 8 * ratio)
                g = int(16 + 22 * ratio)
                b = int(24 + 18 * ratio)
                draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

            # Neon Cyan / Acid Green Radial Burst
            focal_x = int(width * 0.70)
            focal_y = int(height * 0.55)
            glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)

            for rad in range(360, 0, -15):
                alpha = int(50 * (1.0 - (rad / 360.0)))
                glow_draw.ellipse(
                    [focal_x - rad, focal_y - rad, focal_x + rad, focal_y + rad],
                    fill=(0, 255, 170, alpha)
                )
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(35))
            bg = Image.alpha_composite(bg, glow_layer)

            # Tech Grid / Energy Accents
            draw = ImageDraw.Draw(bg)
            for x_offset in range(int(width * 0.5), width, 60):
                draw.line([(x_offset, int(height * 0.3)), (x_offset + 40, height)], fill=(0, 255, 200, 35), width=2)

        # Subtle dark vignette on edges to frame text
        vignette = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        v_draw = ImageDraw.Draw(vignette)
        v_draw.rectangle([0, 0, int(width * 0.5), height], fill=(0, 0, 0, 70))
        vignette = vignette.filter(ImageFilter.GaussianBlur(40))
        bg = Image.alpha_composite(bg, vignette)

        return bg

    # -----------------------------------------------------------------------
    # Typography & Text Overlay
    # -----------------------------------------------------------------------

    def _render_bold_headline(
        self,
        base_img: Image.Image,
        text: str,
        x: int,
        y: int,
        max_w: int,
        max_h: int,
        text_color: Tuple[int, int, int] = (255, 230, 0),
        stroke_color: Tuple[int, int, int] = (0, 0, 0),
        align: str = "left",
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """Render 3-5 word headline with heavy black outline, drop shadow, and auto-sizing."""
        text = text.upper().strip()
        words = text.split()
        if not words:
            return base_img, (x, y, x, y)

        # Dynamic font sizing calculation
        draw_temp = ImageDraw.Draw(base_img)
        font_size = 110
        min_font_size = 48

        while font_size > min_font_size:
            font = self._get_font(font_size)
            lines = self._wrap_words(draw_temp, words, font, max_w)
            line_height = int(font_size * 1.15)
            total_h = len(lines) * line_height
            if len(lines) <= 3 and total_h <= max_h:
                break
            font_size -= 4

        font = self._get_font(font_size)
        lines = self._wrap_words(draw_temp, words, font, max_w)
        line_height = int(font_size * 1.15)
        stroke_w = max(6, int(font_size // 8))

        # 1. Drop shadow layer
        shadow_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_offset = stroke_w + 6

        for idx, line in enumerate(lines):
            line_w = draw_temp.textlength(line, font=font)
            line_x = x if align == "left" else (x + (max_w - line_w) // 2 if align == "center" else x + max_w - line_w)
            line_y = y + idx * line_height

            shadow_draw.text(
                (line_x + shadow_offset, line_y + shadow_offset),
                line,
                font=font,
                fill=(0, 0, 0, 220),
                stroke_width=stroke_w + 4,
                stroke_fill=(0, 0, 0, 220),
            )

        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(stroke_w))
        base_img = Image.alpha_composite(base_img, shadow_layer)

        # 2. Main Text Layer with Sharp Stroke
        text_layer = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_layer)

        for idx, line in enumerate(lines):
            line_w = draw_temp.textlength(line, font=font)
            line_x = x if align == "left" else (x + (max_w - line_w) // 2 if align == "center" else x + max_w - line_w)
            line_y = y + idx * line_height

            # Heavy black border
            text_draw.text(
                (line_x, line_y),
                line,
                font=font,
                fill=(*text_color, 255),
                stroke_width=stroke_w,
                stroke_fill=(*stroke_color, 255),
            )

        base_img = Image.alpha_composite(base_img, text_layer)
        total_bbox = (x, y, x + max_w, y + len(lines) * line_height)
        return base_img, total_bbox

    def _wrap_words(self, draw: ImageDraw.ImageDraw, words: List[str], font: Any, max_w: int) -> List[str]:
        """Wrap words into lines fitting max_w."""
        lines = []
        cur = []
        for w in words:
            candidate = " ".join(cur + [w])
            if draw.textlength(candidate, font=font) <= max_w or not cur:
                cur = cur + [w]
            else:
                lines.append(" ".join(cur))
                cur = [w]
        if cur:
            lines.append(" ".join(cur))
        return lines

    # -----------------------------------------------------------------------
    # Emotion Arrow & Attention Indicators
    # -----------------------------------------------------------------------

    def draw_emotion_arrow(
        self,
        img: Image.Image,
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
        color: Tuple[int, int, int] = (255, 30, 60),
        outline_color: Tuple[int, int, int] = (255, 255, 255),
        thickness: int = 14,
    ) -> Image.Image:
        """Render a bold, high-contrast directional arrow with outline and drop shadow."""
        arrow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(arrow_layer)

        x0, y0 = start_pt
        x1, y1 = end_pt
        dx = x1 - x0
        dy = y1 - y0
        angle = math.atan2(dy, dx)
        head_len = max(35, thickness * 3)

        # Shaft line
        shaft_end_x = x1 - int(head_len * 0.7 * math.cos(angle))
        shaft_end_y = y1 - int(head_len * 0.7 * math.sin(angle))

        # Arrowhead polygon
        left_angle = angle + math.pi * 0.82
        right_angle = angle - math.pi * 0.82
        p1 = (x1, y1)
        p2 = (x1 + int(head_len * math.cos(left_angle)), y1 + int(head_len * math.sin(left_angle)))
        p3 = (shaft_end_x, shaft_end_y)
        p4 = (x1 + int(head_len * math.cos(right_angle)), y1 + int(head_len * math.sin(right_angle)))

        # Outline & shadow pass
        shadow_offset = 6
        draw.line([(x0 + shadow_offset, y0 + shadow_offset), (shaft_end_x + shadow_offset, shaft_end_y + shadow_offset)], fill=(0, 0, 0, 180), width=thickness + 8)
        draw.polygon([(pt[0] + shadow_offset, pt[1] + shadow_offset) for pt in [p1, p2, p3, p4]], fill=(0, 0, 0, 180))

        # White contrasting outer stroke
        draw.line([(x0, y0), (shaft_end_x, shaft_end_y)], fill=(*outline_color, 255), width=thickness + 6)
        draw.polygon([p1, p2, p3, p4], fill=(*outline_color, 255))

        # Inner vibrant color
        draw.line([(x0, y0), (shaft_end_x, shaft_end_y)], fill=(*color, 255), width=thickness)
        # Scaled arrowhead inner
        inner_head_len = head_len - 4
        ip2 = (x1 + int(inner_head_len * math.cos(left_angle)), y1 + int(inner_head_len * math.sin(left_angle)))
        ip4 = (x1 + int(inner_head_len * math.cos(right_angle)), y1 + int(inner_head_len * math.sin(right_angle)))
        draw.polygon([p1, ip2, p3, ip4], fill=(*color, 255))

        return Image.alpha_composite(img, arrow_layer)

    def draw_target_reticle(
        self,
        img: Image.Image,
        center: Tuple[int, int],
        radius: int = 70,
        color: Tuple[int, int, int] = (255, 0, 80),
    ) -> Image.Image:
        """Render a pulsing target reticle or focal ring around key detail."""
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        cx, cy = center

        # Outer glowing circle
        for r in range(radius + 10, radius - 10, -3):
            alpha = int(120 * (1.0 - abs(r - radius) / 10.0))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(*color, alpha), width=3)

        # Crosshairs
        draw.line([(cx - radius - 15, cy), (cx - radius + 15, cy)], fill=(255, 255, 255, 240), width=4)
        draw.line([(cx + radius - 15, cy), (cx + radius + 15, cy)], fill=(255, 255, 255, 240), width=4)
        draw.line([(cx, cy - radius - 15), (cx, cy - radius + 15)], fill=(255, 255, 255, 240), width=4)
        draw.line([(cx, cy + radius - 15), (cx, cy + radius + 15)], fill=(255, 255, 255, 240), width=4)

        return Image.alpha_composite(img, layer)

    # -----------------------------------------------------------------------
    # Reaction Badges & Stickers
    # -----------------------------------------------------------------------

    def draw_reaction_badge(
        self,
        img: Image.Image,
        text: str,
        x: int,
        y: int,
        badge_color: Tuple[int, int, int] = (255, 30, 60),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """Render high-intensity reaction sticker / pill badge (e.g. '😱 INSANE!')."""
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = self._get_font(42)

        text_w = draw.textlength(text, font=font)
        pad_x = 28
        pad_y = 12
        badge_w = int(text_w + pad_x * 2)
        badge_h = int(42 * 1.2 + pad_y * 2)

        # Drop shadow
        draw.rounded_rectangle([x + 4, y + 4, x + badge_w + 4, y + badge_h + 4], radius=16, fill=(0, 0, 0, 160))
        # Outer white stroke
        draw.rounded_rectangle([x - 3, y - 3, x + badge_w + 3, y + badge_h + 3], radius=18, fill=(255, 255, 255, 255))
        # Inner vibrant fill
        draw.rounded_rectangle([x, y, x + badge_w, y + badge_h], radius=16, fill=(*badge_color, 255))

        # Centered badge text with subtle stroke
        draw.text(
            (x + pad_x, y + pad_y - 2),
            text,
            font=font,
            fill=(*text_color, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0, 255),
        )

        return Image.alpha_composite(img, layer)

    # -----------------------------------------------------------------------
    # Watermark
    # -----------------------------------------------------------------------

    def apply_watermark(self, img: Image.Image) -> Image.Image:
        """Overlay channel watermark in corner with crisp transparency."""
        if self.watermark_path.exists():
            try:
                wm = Image.open(self.watermark_path).convert("RGBA")
                # Scale watermark to ~120px width
                wm_w = 120
                wm_h = int(wm.height * (wm_w / wm.width))
                wm = wm.resize((wm_w, wm_h), Image.LANCZOS)
                
                # Apply 75% opacity
                r, g, b, a = wm.split()
                a = a.point(lambda p: int(p * 0.75))
                wm = Image.merge("RGBA", (r, g, b, a))

                # Place in bottom-right with margin
                pos = (img.width - wm_w - 24, img.height - wm_h - 24)
                img.alpha_composite(wm, dest=pos)
                return img
            except Exception as e:
                logger.debug(f"Watermark file load failed: {e}")

        # Fallback text badge
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = self._get_font(24)
        tag = "CLIPMIND"
        draw.text((img.width - 130, img.height - 40), tag, font=font, fill=(255, 255, 255, 140), stroke_width=2, stroke_fill=(0, 0, 0, 180))
        return Image.alpha_composite(img, layer)

    # -----------------------------------------------------------------------
    # A/B Variant Generators
    # -----------------------------------------------------------------------

    def generate_variant_a(
        self,
        topic: str,
        output_path: Path,
        custom_hook: Optional[str] = None,
        width: int = THUMB_WIDTH,
        height: int = THUMB_HEIGHT,
    ) -> ThumbnailVariant:
        """Generate Variant A: Curiosity & Shock Formula (Electric Yellow + Crimson, Left text / Right focal)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rendering Thumbnail Variant A for '{topic}'")

        if not PIL_AVAILABLE:
            with open(output_path, "wb") as f:
                f.write(f"Thumbnail Variant A placeholder for {topic}".encode("utf-8"))
            return ThumbnailVariant(
                variant_id="Variant_A",
                image_path=str(output_path),
                headline=custom_hook or f"THE TRUTH ABOUT {topic}",
                badge_text="😱 99% WRONG!",
                color_palette="Electric Neon Yellow + Crimson",
                layout_style="Left-Text / Right-Focal Reaction",
                has_arrow=True,
                has_badge=True,
                has_watermark=False,
            )

        # 1. Background
        bg = self.generate_background_image(topic, variant_id="A", width=width, height=height)

        # 2. Text formulation (3-5 punchy words)
        if custom_hook:
            headline = custom_hook
        else:
            words = [w for w in topic.split() if len(w) > 3]
            kw = words[0] if words else topic.split()[0]
            headline = f"THE TRUTH ABOUT {kw}"

        # 3. Reaction badge
        bg = self.draw_reaction_badge(bg, "😱 99% WRONG!", x=60, y=70, badge_color=(255, 30, 60))

        # 4. Bold 3-5 word headline overlay (Left side)
        max_text_w = int(width * 0.50)
        max_text_h = int(height * 0.65)
        bg, text_bbox = self._render_bold_headline(
            base_img=bg,
            text=headline,
            x=60,
            y=160,
            max_w=max_text_w,
            max_h=max_text_h,
            text_color=(255, 230, 0),  # Neon Yellow
            stroke_color=(0, 0, 0),
            align="left",
        )

        # 5. Emotion Arrow pointing to right focal subject
        focal_x = int(width * 0.76)
        focal_y = int(height * 0.52)
        arrow_start = (int(width * 0.48), int(height * 0.60))
        bg = self.draw_emotion_arrow(bg, start_pt=arrow_start, end_pt=(focal_x, focal_y), color=(255, 40, 60))

        # 6. Watermark
        bg = self.apply_watermark(bg)

        # 7. Finalize and save
        self._save_thumbnail(bg, output_path)

        ai_prompt = self.generate_ai_prompt(topic, "A")
        return ThumbnailVariant(
            variant_id="A",
            image_path=output_path,
            width=width,
            height=height,
            headline_text=headline,
            reaction_label="😱 99% WRONG!",
            color_palette={"primary": "#FFE600", "accent": "#FF1E3C", "background": "#0C0E14"},
            focal_point="right",
            arrow_target=(focal_x, focal_y),
            ai_prompt=ai_prompt,
            ctr_prediction_score=0.148,
            design_hypothesis="Curiosity Gap: Shock reaction badge with neon yellow headline directs focus to mysterious core discovery.",
        )

    def generate_variant_b(
        self,
        topic: str,
        output_path: Path,
        custom_hook: Optional[str] = None,
        width: int = THUMB_WIDTH,
        height: int = THUMB_HEIGHT,
    ) -> ThumbnailVariant:
        """Generate Variant B: Urgency & Secret Formula (Acid Green + Magenta, Centered text, Target reticle)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Rendering Thumbnail Variant B for '{topic}'")

        if not PIL_AVAILABLE:
            with open(output_path, "wb") as f:
                f.write(f"Thumbnail Variant B placeholder for {topic}".encode("utf-8"))
            return ThumbnailVariant(
                variant_id="Variant_B",
                image_path=str(output_path),
                headline=custom_hook or f"DON'T IGNORE {topic}",
                badge_text="⚠️ SECRET LEAKED",
                color_palette="Acid Green + Neon Cyan + Magenta",
                layout_style="Top-Centered / Target Reticle",
                has_arrow=False,
                has_badge=True,
                has_watermark=False,
            )

        # 1. Background
        bg = self.generate_background_image(topic, variant_id="B", width=width, height=height)

        # 2. Text formulation
        if custom_hook:
            headline = custom_hook
        else:
            words = [w for w in topic.split() if len(w) > 3]
            kw = words[0] if words else topic.split()[0]
            headline = f"{kw} SECRET EXPOSED"

        # 3. Urgency badge
        bg = self.draw_reaction_badge(bg, "🚨 SECRET LEAKED", x=60, y=70, badge_color=(255, 0, 85))

        # 4. Bold headline overlay
        max_text_w = int(width * 0.52)
        max_text_h = int(height * 0.65)
        bg, text_bbox = self._render_bold_headline(
            base_img=bg,
            text=headline,
            x=60,
            y=160,
            max_w=max_text_w,
            max_h=max_text_h,
            text_color=(0, 255, 170),  # Acid Cyan/Green
            stroke_color=(0, 0, 0),
            align="left",
        )

        # 5. Target reticle & arrow
        focal_x = int(width * 0.72)
        focal_y = int(height * 0.55)
        bg = self.draw_target_reticle(bg, center=(focal_x, focal_y), radius=75, color=(255, 0, 85))
        arrow_start = (int(width * 0.46), int(height * 0.68))
        bg = self.draw_emotion_arrow(bg, start_pt=arrow_start, end_pt=(focal_x - 30, focal_y), color=(255, 230, 0))

        # 6. Watermark
        bg = self.apply_watermark(bg)

        # 7. Finalize and save
        self._save_thumbnail(bg, output_path)

        ai_prompt = self.generate_ai_prompt(topic, "B")
        return ThumbnailVariant(
            variant_id="B",
            image_path=output_path,
            width=width,
            height=height,
            headline_text=headline,
            reaction_label="🚨 SECRET LEAKED",
            color_palette={"primary": "#00FFAA", "accent": "#FF0055", "background": "#0E1018"},
            focal_point="center_right",
            arrow_target=(focal_x, focal_y),
            ai_prompt=ai_prompt,
            ctr_prediction_score=0.162,
            design_hypothesis="Urgency & Revelation: Acid green high-contrast text with target reticle primes viewer for exclusive insight.",
        )

    def _save_thumbnail(self, img: Image.Image, out_path: Path) -> Path:
        """Cover-crop and optimize JPEG/PNG under YouTube's 2MB limit."""
        rgb = img.convert("RGB")
        # Save as PNG or JPEG
        if out_path.suffix.lower() == ".png":
            rgb.save(out_path, format="PNG", optimize=True)
            if out_path.stat().st_size > MAX_FILE_BYTES:
                # If PNG exceeds 2MB, fall back to high quality JPEG
                out_path = out_path.with_suffix(".jpg")
                rgb.save(out_path, format="JPEG", quality=90, optimize=True)
        else:
            for q in (95, 90, 85, 78):
                rgb.save(out_path, format="JPEG", quality=q, optimize=True)
                if out_path.stat().st_size <= MAX_FILE_BYTES:
                    break
        return out_path

    # -----------------------------------------------------------------------
    # Master Pair Generation Entry Point
    # -----------------------------------------------------------------------

    def generate_thumbnails(
        self,
        topic: str,
        cycle_id: Optional[str] = None,
        custom_hook_a: Optional[str] = None,
        custom_hook_b: Optional[str] = None,
        target_dir: Optional[Path] = None,
    ) -> ThumbnailResult:
        """Generate Variant A and Variant B for A/B testing."""
        logger.info(f"=== Generating CTR-Maximized Thumbnail Pair for: '{topic}' ===")
        dest_dir = Path(target_dir) if target_dir else self.output_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"cycle_{cycle_id}_" if cycle_id else f"thumb_{int(time.time())}_"
        # Also support canonical names expected by DriveManager: thumbnail_A.png / thumbnail_B.png
        path_a = dest_dir / f"{prefix}thumbnail_A.png"
        path_b = dest_dir / f"{prefix}thumbnail_B.png"

        var_a = self.generate_variant_a(topic=topic, output_path=path_a, custom_hook=custom_hook_a)
        var_b = self.generate_variant_b(topic=topic, output_path=path_b, custom_hook=custom_hook_b)

        result = ThumbnailResult(
            variant_a=var_a,
            variant_b=var_b,
            topic=topic,
            output_dir=dest_dir,
            all_variants=[var_a, var_b],
        )

        logger.info(f"Thumbnails generated: A={var_a.image_path.name} (CTR ~{var_a.ctr_prediction_score:.1%}), B={var_b.image_path.name} (CTR ~{var_b.ctr_prediction_score:.1%})")
        return result


# Module-level convenience function
def generate_thumbnails(
    topic: str,
    cycle_id: Optional[str] = None,
    custom_hook_a: Optional[str] = None,
    custom_hook_b: Optional[str] = None,
    target_dir: Optional[Path] = None,
    config: Optional[Config] = None,
) -> ThumbnailResult:
    """Convenience functional interface for A/B thumbnail generation."""
    designer = ThumbnailDesigner(config=config)
    return designer.generate_thumbnails(
        topic=topic,
        cycle_id=cycle_id,
        custom_hook_a=custom_hook_a,
        custom_hook_b=custom_hook_b,
        target_dir=target_dir,
    )
