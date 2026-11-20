"""Tests for Branding & Logo Overlay (branding_overlay.py)."""
from engine.editing.branding_overlay import build_overlay_filter


def test_build_overlay_filter():
    f_top_right = build_overlay_filter(position="top-right", opacity=0.8, scale_width_px=200)
    assert "scale=200:-2" in f_top_right
    assert "colorchannelmixer=aa=0.80" in f_top_right
    assert "W-w-" in f_top_right

    f_bottom_left = build_overlay_filter(position="bottom-left", opacity=1.0, scale_width_px=150)
    assert "scale=150:-2" in f_bottom_left
    assert "H-h-" in f_bottom_left
