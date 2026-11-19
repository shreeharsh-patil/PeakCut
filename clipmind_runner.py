#!/usr/bin/env python3
"""ClipMind Top-Level CLI & Daemon Runner.

Usage:
    python clipmind_runner.py --wizard        # First-time interactive setup wizard
    python clipmind_runner.py --cycle         # Run a single autonomous cycle
    python clipmind_runner.py --daemon        # Run 12-hour continuous scheduler loop
    python clipmind_runner.py --status        # View API key pools, quotas, and logs
    python clipmind_runner.py --verify-keys   # Verify health of all configured API keys
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from clipmind.cli import main

if __name__ == "__main__":
    main()
