#!/usr/bin/env python3
"""Top-level entrypoint runner for the Anti-Reused Video Automation Pipeline.

Usage Examples:
  # Transform local video into a compliant split-screen 9:16 Short with commentary voiceover
  python run_transform_pipeline.py process -i "input.mp4" -o "output_short.mp4" --layout split_screen --voiceover-text "Here is the critical breakdown of this scene."

  # Validate an existing video against the YouTube YPP compliance score metric
  python run_transform_pipeline.py validate -f "output_short.mp4" --layout split_screen
"""

import sys
import os

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from anti_reused_pipeline.cli import main

if __name__ == "__main__":
    main()
