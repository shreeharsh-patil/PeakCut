import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import batch_runner


def test_load_urls_from_string():
    class Args:
        urls = "https://youtube.com/watch?v=123, https://youtube.com/watch?v=456"
        file = None

    urls = batch_runner.load_urls(Args())
    assert len(urls) == 2
    assert urls[0] == "https://youtube.com/watch?v=123"
    assert urls[1] == "https://youtube.com/watch?v=456"


def test_load_urls_deduplication():
    class Args:
        urls = "https://youtube.com/watch?v=123 https://youtube.com/watch?v=123 https://youtube.com/watch?v=456"
        file = None

    urls = batch_runner.load_urls(Args())
    assert len(urls) == 2
    assert urls == ["https://youtube.com/watch?v=123", "https://youtube.com/watch?v=456"]


def test_load_urls_from_file(tmp_path):
    url_file = tmp_path / "urls.txt"
    url_file.write_text("""
    # This is a comment
    https://youtube.com/watch?v=abc

    https://youtube.com/watch?v=xyz # another comment
    """, encoding="utf-8")

    class Args:
        urls = None
        file = str(url_file)

    urls = batch_runner.load_urls(Args())
    assert len(urls) == 2
    assert urls[0] == "https://youtube.com/watch?v=abc"
    assert "https://youtube.com/watch?v=xyz" in urls[1]
