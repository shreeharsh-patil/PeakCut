import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest

# Make the repo root importable so tests can import the app modules directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests always run the app in self-host (BYOK) mode. app.py freezes
# BILLING_ENABLED at import time and load_dotenv never overrides an existing
# variable, so this must be set here — before any test module imports app — or
# the suite's behavior would depend on the developer's personal .env.
os.environ["BILLING_ENABLED"] = "0"

# Isolate creators and publishing queue to temporary test storage during test runs
_TEST_TEMP_DIR = tempfile.mkdtemp(prefix="test_yt_isolated_")
os.environ["CREATORS_DIR"] = os.path.join(_TEST_TEMP_DIR, "creators")
os.environ["POSTS_QUEUE_PATH"] = os.path.join(_TEST_TEMP_DIR, "posts_queue.json")

from creator_workspaces import CreatorWorkspaceManager
from publishing_queue import reset_publishing_queue

CreatorWorkspaceManager.reset_instance()
reset_publishing_queue()


@pytest.fixture(autouse=True)
def isolate_test_state():
    """Ensure each test runs with isolated CreatorWorkspace and PublishingQueue."""
    yield
    # Clean up any creators or posts created during test in the isolated directory
    creators_dir = Path(os.environ["CREATORS_DIR"])
    if creators_dir.exists():
        for item in creators_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
    posts_file = Path(os.environ["POSTS_QUEUE_PATH"])
    if posts_file.exists():
        try:
            posts_file.write_text("[]\n", encoding="utf-8")
        except OSError:
            pass
