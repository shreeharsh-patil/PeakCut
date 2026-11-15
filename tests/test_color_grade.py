"""Tests for Visual Color Grading (color_grade.py)."""
import pytest

from color_grade import get_color_grade_filter, list_color_grade_presets


def test_list_and_get_presets():
    presets = list_color_grade_presets()
    assert "vibrant_pop" in presets
    assert "warm_cinema" in presets
    assert "moody_dark" in presets
    assert "clean_contrast" in presets
    assert "cyberpunk_neon" in presets

    vf = get_color_grade_filter("vibrant_pop")
    assert "saturation=" in vf
    assert "contrast=" in vf

    vf_warm = get_color_grade_filter("warm_cinema")
    assert "colorbalance=" in vf_warm

    with pytest.raises(ValueError):
        get_color_grade_filter("non_existent_preset")
