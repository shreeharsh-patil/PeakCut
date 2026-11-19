# Patch PyAV 19+ compatibility with faster-whisper (metadata_errors argument removed in av>=19)
try:
    import av
    _orig_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass

import time
import cv2
import subprocess
import argparse
import re
import sys
import threading
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector
except Exception:
    open_video, SceneManager, ContentDetector = None, None, None

import os
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("ULTRALYTICS_SETTINGS", "false")

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None

import os
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable

import yt_dlp

try:
    import mediapipe as mp
except Exception:
    mp = None
# import whisper (replaced by faster_whisper inside function)
from google import genai
from google.genai import types as genai_types

import api_key_rotator
import gemini_worker
import hook_grounding
import layout_picker
import llm_backend
from clip_selection import (build_transcript_windows, clip_count_targets,
                            clip_duration_bounds, snap_clip_to_words,
                            trim_to_best)
from ffmpeg_utils import (video_encode_args, audio_encode_args, QUALITY,
                          QUALITY_FAST, METADATA_SCRUB)
from dotenv import load_dotenv
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO = 9 / 16

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst). In the descriptions, ALWAYS include a CTA like "Follow me and comment X and I'll send you the workflow" (especially if discussing an n8n workflow):
{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "video_description_for_tiktok": "<description for TikTok oriented to get views>",
      "video_description_for_instagram": "<description for Instagram oriented to get views>",
      "video_title_for_youtube_short": "<title for YouTube Short oriented to get views 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay (max 10 words) with 1-2 fitting emojis. MUST BE IN THE SAME LANGUAGE AS THE VIDEO TRANSCRIPT. Examples: 'POV: You realized... 😳', 'Did you know? 🤯', 'Stop doing this! 🚫'>"
    }}
  ]
}}
"""

# Load the YOLO model once (Keep for backup or scene analysis if needed)
model = None
if YOLO is not None:
    try:
        yolo_path = os.environ.get("YOLO_MODEL_PATH")
        if not yolo_path:
            for cand in ["models/yolov8n.pt", "assets/yolov8n.pt", "yolov8n.pt"]:
                if os.path.exists(cand):
                    yolo_path = cand
                    break
            else:
                yolo_path = "models/yolov8n.pt"
        model = YOLO(yolo_path)
    except Exception as e:
        print(f"⚠️ YOLO model init notice: {e}")

# --- MediaPipe Setup ---
mp_face_detection = None
face_detection = None
if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    except Exception as e:
        print(f"⚠️ MediaPipe face detection init notice: {e}")

# Consecutive detections a large target move must survive before the camera
# follows it (see SmoothedCameraman.update_target). Env-overridable so the
# damping can be dialled back without a deploy; 1 restores the old behaviour.
JUMP_CONFIRM_FRAMES = max(int(os.environ.get("JUMP_CONFIRM_FRAMES", "3")), 1)

# Reset the tracker and the cameraman's damping at every scene cut, so the
# first face found in the new shot is framed instantly instead of being treated
# as a suspicious "jump" from the previous shot's subject (see
# SmoothedCameraman.begin_scene). 0 restores the old behaviour.
SCENE_CUT_RESET = os.environ.get("SCENE_CUT_RESET", "1") != "0"


class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """
    def __init__(self, output_width, output_height, video_width, video_height, aspect_ratio=ASPECT_RATIO):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        if not aspect_ratio or aspect_ratio <= 0:
            aspect_ratio = ASPECT_RATIO
        self.aspect_ratio = aspect_ratio

        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / aspect_ratio)

        # Enforce even dimensions for encoder compatibility
        self.crop_width = max(2, self.crop_width - (self.crop_width % 2))
        self.crop_height = max(2, self.crop_height - (self.crop_height % 2))
             
        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25

        # A target that teleports further than the safe zone in one detection is
        # far more often a detector error — a second face, a false positive, a
        # box snapping to a different body part — than a person who actually
        # moved that far. Committing to it immediately is what made the camera
        # swing: measured on real user footage, 22% of target updates jumped
        # more than the entire safe zone. So a big move has to REPEAT this many
        # times before the camera follows it; a wrong reading disappears on the
        # next detection and never moves the frame.
        #
        # The cost is latency on a genuinely fast move: at DETECT_STRIDE=4 and
        # 30fps, three confirmations is ~0.4s. That reads as an operator being
        # unhurried, which is the look we want, and it is far cheaper than the
        # whip-panning it replaces.
        #
        # Measured over 262s of TRACK footage from two real user videos
        # (26-jul-2026), confirm=1 -> 3: in-scene reversals 0.41/s -> 0.13/s
        # (-69%), camera travel 91px/s -> 60px/s (-34%). Per scene, 54 of 84 get
        # calmer and 23 are unchanged — but 7 get BUSIER, up to 59 -> 108px/s,
        # because committing later can leave the camera further to travel. Net
        # strongly positive, not universally so.
        self.jump_confirm_frames = JUMP_CONFIRM_FRAMES
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = False

    def begin_scene(self):
        """Forget the previous shot's subject at a scene cut.

        The jump damping above exists to reject detector noise INSIDE a shot.
        Across a cut it does the opposite of what is wanted: the new shot's face
        is (by construction) far from the old target, so it was held back for
        JUMP_CONFIRM_FRAMES detections and the camera then panned towards it at
        pan speed. On a real two-camera podcast (24-aug-2026) that showed up as
        a headless torso for 1.5s after every cut while the frame slid over to
        the speaker. The snap at the scene's first frame did not help: the
        target it snapped to was still the previous shot's.

        So: drop any pending jump, and cut (rather than pan) to the first target
        accepted in the new shot.
        """
        self._pending_target = None
        self._pending_count = 0
        self._snap_pending = True

    def update_target(self, face_box):
        """Update the target centre from a detection, ignoring lone big jumps."""
        if not face_box:
            return
        x, y, w, h = face_box
        new_center = x + w / 2

        if self._snap_pending:
            self._snap_pending = False
            self._pending_target = None
            self._pending_count = 0
            self.target_center_x = new_center
            self.current_center_x = new_center
            return

        if abs(new_center - self.target_center_x) > self.safe_zone_radius:
            # Same big move as last time? Count it. Otherwise start counting
            # afresh — two contradictory outliers must not confirm each other.
            if (self._pending_target is not None
                    and abs(new_center - self._pending_target) <= self.safe_zone_radius):
                self._pending_count += 1
            else:
                self._pending_target = new_center
                self._pending_count = 1
            if self._pending_count < self.jump_confirm_frames:
                return  # not convinced yet — hold the frame

        self._pending_target = None
        self._pending_count = 0
        self.target_center_x = new_center
    
    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            
            # SIMPLIFIED LOGIC:
            # 1. Is the target outside the safe zone?
            if abs(diff) > self.safe_zone_radius:
                # 2. If yes, move towards it slowly (Linear Speed)
                # Determine direction
                direction = 1 if diff > 0 else -1
                
                # Speed: 2 pixels per frame (Slow pan)
                # If the distance is HUGE (scene change or fast movement), speed up slightly
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0 # Fast re-frame
                else:
                    speed = 3.0  # Slow, steady pan
                
                self.current_center_x += direction * speed
                
                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x
            
            # If inside safe zone, DO NOTHING (Stationary Camera)
                
        # Clamp center
        half_crop = self.crop_width / 2
        
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop
            
        x1 = int(round(self.current_center_x - half_crop))
        x1 = max(0, min(x1, self.video_width - self.crop_width))
        x2 = min(self.video_width, x1 + self.crop_width)
        
        half_crop_y = self.crop_height / 2
        y1 = int(round(self.video_height / 2.0 - half_crop_y))
        y1 = max(0, min(y1, self.video_height - self.crop_height))
        y2 = min(self.video_height, y1 + self.crop_height)
        
        return x1, y1, x2, y2

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker
        
        # Hyperparameters
        self.stabilization_threshold = stabilization_frames # Frames needed to confirm a new speaker
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000
        
        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'center': x, 'last_frame': 123}]

    def reset(self):
        """Forget every speaker at a scene cut.

        Identity, hysteresis and the switch cooldown are all about continuity
        within a shot. After a cut none of it applies: the sticky x3 bonus and
        the cooldown were holding the previous shot's speaker (returning None)
        for up to 30 frames while a new face sat unframed.
        """
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.last_switch_frame = -1000
        self.known_faces = []

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []
        
        # 1. Match faces to known IDs (simple distance tracking)
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            
            best_match_id = -1
            min_dist = width * 0.15 # Reduced matching radius to avoid jumping in groups
            
            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30: # Forgot faces older than 1s (was 2s)
                    continue
                    
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']
            
            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            
            # Update known face
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            
            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0 # Sticky factor
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            # New person. The cooldown must hold whether or not the current
            # speaker happens to be detected in THIS frame.
            #
            # It used to fall through and switch when the active speaker was
            # missing from the candidate list — a blink, a head turn or one
            # motion-blurred frame was enough. That is precisely when the
            # cooldown is needed, so it only ever fired when it wasn't: 3 of 7
            # target switches measured on a 12s clip (25-jul-2026) jumped the
            # cooldown this way, and every jump drags the camera across frame.
            #
            # Returning None holds instead: the caller only calls
            # update_target() on a truthy box, so the camera keeps its current
            # target and finishes whatever move it was making. The hold is
            # bounded by the cooldown itself — once it expires, a speaker who
            # really did leave the shot is switched away from normally.
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                return old_cand['box'] if old_cand else None

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

# Detectors never need full-resolution frames: MediaPipe returns relative
# coords and YOLO boxes are scaled back up. Running them on a ≤640px copy cuts
# per-frame preprocessing cost hard, which is what dominates CPU-only renders.
DETECT_MAX_WIDTH = 640
# The global MediaPipe graph and YOLO model are NOT thread-safe; clips render
# in parallel, so every inference goes through this lock. Contention is small
# (a few ms per call) — the ffmpeg renders are where the parallel time goes.
DETECT_LOCK = threading.Lock()
# Detect every Nth frame; SmoothedCameraman interpolates between updates.
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "4")), 1)
# YOLO fallback (no face found) is far heavier than MediaPipe — extra throttle.
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2


def _detection_frame(frame):
    """Downscaled copy for detectors. Returns (small_frame, scale) with
    scale mapping small-frame pixel coords back to the original frame."""
    h, w = frame.shape[:2]
    if w <= DETECT_MAX_WIDTH:
        return frame, 1.0
    scale = w / DETECT_MAX_WIDTH
    small = cv2.resize(frame, (DETECT_MAX_WIDTH, max(int(h / scale), 2)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    Boxes are in ORIGINAL frame coordinates (detection runs downscaled;
    MediaPipe's relative coords make the mapping exact).
    """
    if face_detection is None:
        height, width, _ = frame.shape
        w = int(width * 0.4)
        h = int(height * 0.5)
        x = (width - w) // 2
        y = int(height * 0.15)
        return [{'box': [x, y, w, h], 'score': w * h}]

    height, width, _ = frame.shape
    small, _scale = _detection_frame(frame)
    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with DETECT_LOCK:
        results = face_detection.process(rgb_frame)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        candidates.append({
            'box': [x, y, w, h],
            'score': w * h # Area as score
        })
            
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation, in
    ORIGINAL frame coordinates (inference runs on a downscaled copy).
    """
    if model is None:
        return None
    small, scale = _detection_frame(frame)
    # Use the globally loaded model
    with DETECT_LOCK:
        results = model(small, verbose=False, classes=[0]) # class 0 is person

    if not results:
        return None

    best_box = None
    max_area = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i * scale) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
                
    return best_box

def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Fill Height)
    # Crop center to aspect ratio
    bg_scale = output_height / orig_h
    bg_w = int(orig_w * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, output_height), interpolation=cv2.INTER_LINEAR)

    # Crop center of background
    start_x = (bg_w - output_width) // 2
    if start_x < 0: start_x = 0
    background = bg_resized[:, start_x:start_x+output_width]
    if background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    # Blur background: blur at quarter resolution and scale back up — visually
    # identical for a defocused backdrop, an order of magnitude cheaper than a
    # 51px Gaussian at full size.
    small_bg = cv2.resize(background, (max(output_width // 4, 2), max(output_height // 4, 2)),
                          interpolation=cv2.INTER_AREA)
    small_bg = cv2.GaussianBlur(small_bg, (13, 13), 0)
    background = cv2.resize(small_bg, (output_width, output_height),
                            interpolation=cv2.INTER_LINEAR)

    # 2. Foreground (Fit Width)
    scale = output_width / orig_w
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (output_width, fg_h), interpolation=cv2.INTER_LINEAR)

    # A source taller than the output fills the width at a height that does not
    # fit: centre-crop it instead of indexing the frame with a negative offset,
    # which raises rather than renders.
    if fg_h > output_height:
        top = (fg_h - output_height) // 2
        foreground = foreground[top:top + output_height, :]
        fg_h = output_height

    # 3. Overlay
    y_offset = (output_height - fg_h) // 2

    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, :] = foreground
    
    return final_frame

# NOTE: a "route text-heavy scenes to GENERAL" rule was tried here and removed
# on 26-jul-2026. The problem it targets is real — a screencast that happens to
# contain one face gets cropped to the face and its headlines come out cut
# mid-word — but edge density is the wrong signal for it. Measured: a
# constructed talking-head-beside-a-chart scored 0.012 while the SAME shot
# without the panels scored 0.029, because a flat panel of text has far fewer
# edges than ordinary scene detail. Canny measures visual busyness, not text.
# A real fix needs an actual text detector (MSER/EAST) validated against clips
# that contain the failure mode; this corpus has almost none.


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []

    if not cap.isOpened():
        return ['TRACK'] * len(scenes)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        s_f, e_f = start.get_frames(), end.get_frames()
        # Sample 5 frames spread across the scene, clamped inside it (the old
        # start+5/end-5 samples landed outside scenes shorter than ~10 frames).
        margin = min(2, max(0, (e_f - s_f - 1) // 2))
        frames_to_check = sorted(set(
            int(round(f)) for f in np.linspace(s_f + margin, e_f - 1 - margin, 5)
        ))

        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue

            # Near-black frames (fades, cut-to-black) carry no faces and used
            # to drag single-person scenes into GENERAL. Skip them.
            if frame.mean() < 16:
                continue

            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))

        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)

        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)

        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')

    cap.release()

    # Hysteresis: a short scene whose two neighbors agree on the opposite
    # strategy is almost always a sampling miss (profile face, insert shot).
    # Each TRACK<->GENERAL flip is a full on-screen layout change, so flapping
    # is worse than an occasional wrong-but-stable choice.
    max_flip_frames = int(2.0 * fps)
    for i in range(1, len(strategies) - 1):
        dur = scenes[i][1].get_frames() - scenes[i][0].get_frames()
        if (dur < max_flip_frames
                and strategies[i - 1] == strategies[i + 1] != strategies[i]):
            strategies[i] = strategies[i - 1]

    return strategies

def detect_scenes(video_path):
    import scene_detection
    return scene_detection.detect_scenes(video_path)

def get_video_resolution(video_path):
    probe = cv2.VideoCapture(video_path)
    try:
        if probe.isOpened():
            w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
    finally:
        probe.release()

    # Fallback to ffprobe if cv2 failed or returned 0
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=s=x:p=0",
            str(video_path)
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and "x" in out.stdout:
            parts = out.stdout.strip().split("x")
            return int(parts[0]), int(parts[1])
    except Exception:
        pass

    raise IOError(f"cannot open video or determine resolution: {video_path}")


# Byte budget for the sanitized video title used as the stem of every derived
# file. Filesystems cap a name in BYTES (255 on ext4), not characters, and the
# pipeline decorates this stem: "_clip_10.mp4" (12), "subtitled_<ts>_" (21),
# "hooked_<ts>_" (18), "temp_hook_<hex8>_" (19), "autosubs_<ts>_" + ".ass" (24).
# Budgeting 120 bytes leaves room for all of them stacked (worst chain:
# subtitled_<ts>_hooked_<ts>_<stem>_clip_NN.mp4 ≈ 171 bytes) under the limit.
#
# The old cap was 100 CHARACTERS, which is 300 bytes of Bengali or Arabic — over
# the limit before any decoration. It surfaced as OSError 36 killing the hook
# endpoint in prod on 26-jul-2026.
MAX_TITLE_BYTES = 120


def truncate_bytes(text, max_bytes):
    """Trim ``text`` to a byte budget without splitting a multi-byte character."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", "ignore")


def sanitize_filename(filename):
    """Remove invalid characters from filename and bound it for the filesystem."""
    # "canción" has two Unicode spellings: a precomposed ó (NFC) or an o plus a
    # combining acute (NFD). yt-dlp hands over titles in either, and the name
    # becomes the clip file, the R2 key and the URL path. Measured 24-ago-2026:
    # a key carrying the combining form is fetchable by a <video> element but a
    # fetch() of the same URL comes back 503, which broke the download button on
    # every clip with a Spanish title. Normalising here fixes the whole chain at
    # its source, and is a no-op for the ASCII names that already worked.
    filename = unicodedata.normalize('NFC', filename)
    filename = re.sub(r'[<>:"/\\|?*#]', '', filename)
    filename = filename.replace(' ', '_')
    return truncate_bytes(filename, MAX_TITLE_BYTES)


def is_youtube_url(url):
    """True for the hosts the proxy chain exists for. Anything else (a CDN
    mp4, tmpfiles/catbox, an R2 link) has no IP ban to dodge and downloads
    5-10x faster from the server's own IP than through the ISP proxies."""
    try:
        from security_utils import normalize_media_url
        url = normalize_media_url(url)
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return True
    return host.endswith(("youtube.com", "youtu.be", "youtube-nocookie.com", "googlevideo.com"))


def plan_download_attempts(direct_first, statics, paid, have_hd, youtube=True):
    """Ordered (label, capped, proxy) download plan — pure, unit-tested.

    ``youtube=False`` (a direct file URL): the server's own IP first, then one
    static proxy as the only fallback; the paid per-GB proxy is never used.

    Cheapest bandwidth first: the server's own IP, then the flat-rate static
    ISP proxies (uncapped 1080p, free bytes), then the per-GB paid proxy
    (720p cost cap), and last the conservative fallback strategy through the
    paid proxy (or a static/direct when no paid proxy is configured).
    ``capped`` marks attempts whose bytes are billed per GB."""
    if not youtube:
        plan = [('direct', False, None)]
        if statics:
            plan.append(('static-fallback', False, statics[0]))
        return plan
    plan = []
    if direct_first:
        plan.append(('HD-direct', False, None))
    if have_hd:
        for i, s in enumerate(statics):
            plan.append((f'HD-static{i + 1}', False, s))
    if statics and paid:
        # The conservative clients (tv_embed/android) through a FREE static,
        # before any per-GB attempt: YouTube serves a fake "Video unavailable"
        # to the web/HD client from datacenter-ISP ranges on some videos while
        # the fallback clients pass on the very same IPs (verified 4-sep-2026,
        # all three statics, three countries). Costs nothing and the paid path
        # was capped to 720p anyway, so there is no quality trade.
        plan.append(('fallback-static', False, statics[0]))
    if have_hd:
        plan.append(('HD', bool(paid), paid))
    plan.append(('fallback', bool(paid),
                 paid if paid else (statics[0] if statics else None)))
    return plan


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    from security_utils import assert_public_url, normalize_media_url
    url = normalize_media_url(url)
    assert_public_url(url)
    # Throwaway hosts agents fall back to (tmpfiles.org) hand out short-lived
    # signed links; refresh through the host's page so yt-dlp gets the file.
    import file_hosts
    url = file_hosts.resolve(url)

    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = '/app/cookies.txt'
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    if cookies_env:
        print("🍪 Found YOUTUBE_COOKIES env var, creating cookies file inside container...")
        try:
            with open(cookies_path, 'w', encoding="utf-8") as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                 # Never print file CONTENT here: with a headerless cookies
                 # blob this would leak live YouTube session cookies to logs.
                 print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        cookies_path = None
        print("⚠️ YOUTUBE_COOKIES env var not found.")
    
    # Optional HTTP proxy. Set PROXY_URL to route downloads through it; unset
    # (self-host) goes direct as before.
    _proxy = os.environ.get("PROXY_URL", "").strip() or None
    if _proxy:
        print("🌐 Using proxy for download.")

    # Flat-rate static ISP proxies (STATIC_PROXY_URLS, comma-separated), tried
    # BEFORE the per-GB proxy: dedicated IPs with unlimited traffic, so their
    # bandwidth costs nothing per job and carries no 720p cost cap. Rotated per
    # job to spread load (and YouTube's attention) across the pool. PROXY_URL
    # stays the paid last resort — with STATIC_PROXY_URLS unset the behavior is
    # byte-identical to before.
    _statics = [p.strip() for p in
                os.environ.get("STATIC_PROXY_URLS", "").split(",") if p.strip()]
    if _statics:
        import random as _random
        k = _random.randrange(len(_statics))
        _statics = _statics[k:] + _statics[:k]
        print(f"🌐 {len(_statics)} static ISP proxies configured.")

    # Two download strategies, tried in order so a break in the HD path degrades
    # gracefully instead of failing the whole job: an HD attempt first, then a
    # conservative fallback (also the only strategy for self-host).
    _bgutil_http = os.environ.get("BGUTIL_BASE_URL", "").strip()
    _bgutil_script = os.environ.get("BGUTIL_SCRIPT_PATH", "").strip()
    if _bgutil_http:
        hd_args = {'youtubepot-bgutilhttp': {'base_url': [_bgutil_http]}}
    elif _bgutil_script:
        hd_args = {'youtubepot-bgutilscript': {'script_path': [_bgutil_script]}}
    else:
        hd_args = None
    fallback_args = {
        'youtube': {
            'player_client': ['tv_embed', 'android', 'mweb', 'web'],
            'player_skip': ['webpage', 'configs'],
        }
    }

    # Cap at 720p ONLY when the bytes actually go through the PER-GB paid proxy
    # — that cap exists to control bandwidth cost, and the direct attempt and
    # the flat-rate static proxies have none.
    #
    # This is per-attempt on purpose. Deciding it once from `_proxy` capped the
    # DIRECT attempt too, so with DIRECT_FIRST=1 (which serves most downloads)
    # every YouTube source arrived at 720p and, since the reframe inherits the
    # source height, 80% of delivered clips came out 406x720 (audited 25-jul-2026).
    def _hd_fmt_for(capped):
        if capped:
            return ('bestvideo[vcodec^=avc1][height<=720][ext=mp4]+bestaudio[ext=m4a]/'
                    'bestvideo[vcodec^=avc1][height<=720]+bestaudio/'
                    'best[height<=720][ext=mp4]/best[height<=720]/best')
        return ('bestvideo[vcodec^=avc1][height<=1080][ext=mp4]+bestaudio[ext=m4a]/'
                'bestvideo[vcodec^=avc1][height<=1080]+bestaudio/'
                'best[height<=1080][ext=mp4]/best[ext=mp4]/best')
    fallback_fmt = 'best[ext=mp4]/best'

    def _base_opts(extractor_args, proxy):
        return {
            'quiet': False, 'verbose': True, 'no_warnings': False,
            'cookiefile': cookies_path if cookies_path else None,
            'proxy': proxy, 'socket_timeout': 30, 'retries': 10, 'fragment_retries': 10,
            'nocheckcertificate': True, 'cachedir': False,
            'extractor_args': extractor_args,
            'http_headers': {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                ),
            },
        }

    # Wire bytes actually pulled through the (paid) proxy, summed across
    # fragments/streams. Reported to app.py via the PROXY_BYTES= line below.
    _dl_bytes = {"total": 0, "partial": 0}

    def _progress_hook(d):
        if d.get('status') == 'downloading':
            # Bytes of a fragment still in flight: a failed attempt has
            # already paid for these even though 'finished' never fires.
            _dl_bytes["partial"] = int(d.get('downloaded_bytes') or 0)
        elif d.get('status') == 'finished':
            _dl_bytes["partial"] = 0
            _dl_bytes["total"] += int(d.get('total_bytes')
                                      or d.get('total_bytes_estimate')
                                      or d.get('downloaded_bytes') or 0)

    def _attempt(extractor_args, fmt, proxy):
        _dl_bytes["total"] = 0
        _dl_bytes["partial"] = 0
        with yt_dlp.YoutubeDL(_base_opts(extractor_args, proxy)) as ydl:
            info = ydl.extract_info(url, download=False)
        sanitized = sanitize_filename(info.get('title', 'youtube_video'))
        expected = os.path.join(output_dir, f'{sanitized}.mp4')
        if os.path.exists(expected):
            os.remove(expected)

        if info.get('live_status') == 'is_upcoming':
            raise ValueError(f"YouTube Live stream '{info.get('title')}' is upcoming and has not started yet.")

        is_live = bool(info.get('is_live') or info.get('live_status') == 'is_live')
        if is_live:
            live_sec = int(os.environ.get("LIVE_RECORD_SECONDS", "900"))
            print(f"🔴 Active YouTube Live Stream detected: '{info.get('title')}'")
            print(f"⏱️ Capturing {live_sec} seconds ({live_sec/60:.1f} mins) of live stream...")

            # Extract playable stream manifest/URL (pairing video and audio if separate)
            candidates = info.get("requested_formats")
            stream_urls = []
            if candidates:
                stream_urls = [f.get("url") for f in candidates if isinstance(f, dict) and f.get("url")]
            if not stream_urls and info.get("url"):
                stream_urls = [info.get("url")]
            if not stream_urls and info.get("formats"):
                fmts = [f for f in info.get("formats", []) if isinstance(f, dict) and f.get("url")]
                combined = [f for f in fmts if f.get("vcodec") != "none" and f.get("acodec") != "none"]
                if combined:
                    best = max(combined, key=lambda f: f.get("height") or 0)
                    stream_urls = [best.get("url")]
                else:
                    videos = [f for f in fmts if f.get("vcodec") != "none"]
                    audios = [f for f in fmts if f.get("acodec") != "none"]
                    if videos:
                        stream_urls.append(max(videos, key=lambda f: f.get("height") or 0).get("url"))
                    if audios:
                        stream_urls.append(max(audios, key=lambda f: f.get("abr") or 0).get("url"))
                    if not stream_urls and fmts:
                        stream_urls = [fmts[-1].get("url")]

            if stream_urls:
                ff_cmd = [
                    "ffmpeg", "-y",
                    "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10",
                ]
                for u in stream_urls:
                    ff_cmd.extend(["-i", str(u)])
                if len(stream_urls) > 1:
                    ff_cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
                else:
                    ff_cmd.extend(["-map", "0:v:0", "-map", "0:a?"])
                ff_cmd.extend([
                    "-t", str(live_sec),
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    expected
                ])
                res = subprocess.run(ff_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if res.returncode == 0 and os.path.exists(expected) and os.path.getsize(expected) > 1000:
                    print(f"✅ Live stream segment captured: {expected} ({os.path.getsize(expected)/(1024*1024):.1f} MB)")
                    return sanitized
                else:
                    err = res.stderr.decode('utf-8', errors='ignore') if res.stderr else "unknown error"
                    print(f"⚠️ FFmpeg stream capture warning ({err[:150]}), falling back to yt-dlp...")

        dl_opts = {
            **_base_opts(extractor_args, proxy),
            'format': fmt,
            'outtmpl': os.path.join(output_dir, f'{sanitized}.%(ext)s'),
            'merge_output_format': 'mp4', 'overwrites': True,
            'progress_hooks': [_progress_hook],
        }
        if is_live:
            dl_opts['live_from_start'] = False
            dl_opts['downloader'] = 'ffmpeg'
            dl_opts['downloader_args'] = {'ffmpeg': ['-t', str(live_sec)]}
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            ydl.download([url])
        return sanitized

    # DIRECT_FIRST=1: try the server's own IP before spending proxy bandwidth.
    # Needs cookies + a PO-token provider — without both, YouTube flags the
    # datacenter IP after the first request (verified in prod, 21-jul-2026).
    _direct_first = (os.environ.get("DIRECT_FIRST", "").strip() == "1"
                     and (_proxy or _statics) and hd_args and cookies_path)

    attempts = [
        (label,
         fallback_args if label.startswith('fallback') else hd_args,
         fallback_fmt if label.startswith('fallback') else _hd_fmt_for(capped),
         proxy)
        for label, capped, proxy in plan_download_attempts(
            _direct_first, _statics, _proxy, bool(hd_args), youtube=is_youtube_url(url))
    ]
    if not is_youtube_url(url):
        print("🌐 Direct file URL: downloading from the server's own IP (no proxy).")

    sanitized_title = None
    last_err = None
    used_proxy = False
    # Every attempt, with its bytes and failure text: printed as PROXY_ROUTE=
    # below so app.py can keep a durable trail of WHY a job reached the paid
    # proxy (the container log rotates within the hour; see cloud/proxy_ledger).
    attempt_log = []
    for label, ea, fmt, proxy in attempts:
        # A 403 on the media fetch is usually transient: the googlevideo URL is
        # bound to the IP that extracted it, and the residential proxy rotates
        # its exit IP between requests. Retrying re-extracts and usually lands
        # on a consistent IP (3 of 62 downloads hit this on 22-jul-2026).
        for retry in range(2):
            try:
                print(f"📥 Download attempt: {label}" + (f" (retry {retry})" if retry else ""))
                sanitized_title = _attempt(ea, fmt, proxy)
                # Only bytes through the PER-GB proxy cost money; direct and
                # the flat-rate static proxies are free bandwidth for the
                # monthly counter's purposes.
                used_proxy = proxy is not None and proxy == _proxy
                attempt_log.append({"label": label, "ok": True,
                                    "bytes": _dl_bytes["total"] + _dl_bytes["partial"],
                                    "paid": used_proxy})
                print(f"✅ Download succeeded ({label}).")
                break
            except Exception as e:
                last_err = e
                attempt_log.append({"label": label, "ok": False,
                                    "bytes": _dl_bytes["total"] + _dl_bytes["partial"],
                                    "paid": proxy is not None and proxy == _proxy,
                                    "error": str(e)[:300]})
                print(f"⚠️  Download attempt '{label}' failed: {str(e)[:200]}")
                err_str = str(e).lower()
                retryable = any(k in err_str for k in [
                    '403', 'forbidden', 'timeout', 'timed out', 'connection reset',
                    '503', '429', 'remotedisconnected', 'broken pipe', 'connection refused'
                ])
                if not retryable or retry == 1:
                    break
                time.sleep(3)
        if sanitized_title is not None:
            break

    if sanitized_title is None:
        import sys
        error_msg = f"""
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED (all strategies)
❌ ================================================================= ❌
REASON: YouTube blocked the request or the download tooling is out of date.
👇 SOLUTION FOR USER: download the video manually and use the 'Upload Video' tab.
Technical Details: {str(last_err)}
"""
        print(error_msg, file=sys.stdout)
        print(error_msg, file=sys.stderr)
        sys.stdout.flush(); sys.stderr.flush()
        time.sleep(0.5)
        raise last_err

    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if not os.path.exists(downloaded_file):
        for ext in ('.mp4', '.mkv', '.webm', '.mov', '.ts'):
            for f in os.listdir(output_dir):
                if f.startswith(sanitized_title) and f.endswith(ext):
                    candidate = os.path.join(output_dir, f)
                    if ext == '.mp4':
                        downloaded_file = candidate
                    else:
                        remuxed = os.path.join(output_dir, f'{sanitized_title}.mp4')
                        remux_cmd = ['ffmpeg', '-y', '-i', candidate, '-c', 'copy', '-movflags', '+faststart', remuxed]
                        r = subprocess.run(remux_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        if r.returncode == 0 and os.path.exists(remuxed):
                            try:
                                os.remove(candidate)
                            except OSError:
                                pass
                            downloaded_file = remuxed
                    break
            if os.path.exists(downloaded_file):
                break

    # Paid bytes across EVERY attempt that used the per-GB proxy, failed ones
    # included: a paid attempt that died after three 10 MB fragments was
    # billed for them even though a later attempt won.
    paid_bytes = sum(int(a.get("bytes") or 0) for a in attempt_log if a.get("paid"))
    print("PROXY_ROUTE=" + json.dumps({
        "winner": attempt_log[-1]["label"] if attempt_log and attempt_log[-1].get("ok") else None,
        "paid_bytes": paid_bytes,
        "attempts": attempt_log,
    }, ensure_ascii=False))
    if paid_bytes:
        # Machine-parseable marker consumed by app.py's log reader for the
        # monthly proxy-bandwidth counter. Not shown to clients (log filter).
        print(f"PROXY_BYTES={paid_bytes}")
    print(f"✅ Video downloaded in {time.time() - step_start_time:.2f}s: {downloaded_file}")
    return downloaded_file, sanitized_title

def finalize_clip_passthrough(input_video, final_output_video):
    """Keep the clip's native framing (for horizontal/16:9 output).

    The input is the freshly encoded cut, so a stream-copy remux is enough to
    add +faststart — re-encoding here would only cost time and quality.
    """
    if os.path.exists(final_output_video):
        os.remove(final_output_video)
    print(f"🎬 Passthrough (native framing): {input_video}")
    cmd = [
        'ffmpeg', '-y', '-i', input_video,
        '-c', 'copy', *METADATA_SCRUB, '-movflags', '+faststart',
        final_output_video,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
    print(f"✅ Clip saved to {final_output_video}")
    return True


def auto_caption_clip(clip_path, transcript, clip_start, clip_end, split_ranges=None):
    """Burn the default caption style onto a finished clip.

    ``split_ranges``: (start, end) stretches, in clip seconds, rendered with
    the SPLIT layout; captions there sit on the seam between the two speakers
    instead of the bottom. None reads the render's own sidecar next to
    ``clip_path`` (layout_ranges), which is where recut hands it over.

    Captions are mandatory for short-form to land, but they were opt-in behind a
    modal and only 9% of delivered clips ever got them (prod audit, 25-jul-2026).
    So every clip now ships captioned by default.

    The captioned file is written ALONGSIDE the clip as
    ``subtitled_<ts>_<clip>.mp4`` — the same convention /api/subtitle uses — so
    the untouched original stays on disk and re-styling from the modal replaces
    the captions instead of burning a second layer over them.

    Returns the captioned path, or None when captions were skipped (silent
    video, no words in range, AUTO_CAPTIONS=0, or any failure — a caption
    problem must never cost the user the clip they already paid for).
    """
    if os.environ.get("AUTO_CAPTIONS", "1").strip() == "0":
        return None
    if not transcript or not transcript.get('segments'):
        return None  # silent video: nothing to caption
    try:
        import subtitles as _subs
        style = _subs.AUTO_CAPTION_STYLE
        output_dir = os.path.dirname(clip_path)
        stem = os.path.basename(clip_path)
        generation_id = int(time.time())
        # The output name MUST stay exactly "subtitled_<ts>_<clip filename>":
        # the modal's walk-back and _canonical_clip_file both reconstruct the
        # clean original from it, so trimming the stem here would orphan the
        # pair. Length is bounded upstream instead, by MAX_TITLE_BYTES at
        # download time. A legacy clip whose name predates that budget can still
        # overflow — that raises OSError 36, which the except below turns into
        # "ship the clip uncaptioned" rather than a broken filename.
        # The .ass path is interpolated INTO an ffmpeg filter string
        # (-vf ass='...'), where a literal apostrophe closes the quote and
        # breaks the filter. Titles carry apostrophes constantly in English
        # ("Earth's", "Don't"), so this name must stay free of the clip stem —
        # which is exactly why /api/subtitle has always used a neutral
        # "subs_<i>_<ts>.ass". Deriving it from the stem silently cost captions
        # on every apostrophe title until 29-jul-2026.
        #
        # The OUTPUT name still carries the stem, and must: the modal's
        # walk-back and _canonical_clip_file reconstruct the clean original
        # from it. That one is only ever passed as an argv element, never
        # inside a filter string, so quoting never applies to it.
        # Unique per clip, not just per second: clips render in parallel
        # (CLIP_WORKERS), so a bare timestamp would collide and let one clip
        # burn another's captions.
        ass_path = os.path.join(
            output_dir, f"autosubs_{generation_id}_{uuid.uuid4().hex[:8]}.ass")
        out_path = os.path.join(output_dir, f"subtitled_{generation_id}_{stem}")

        try:
            if split_ranges is None:
                import layout_ranges as _layouts
                split_ranges = _layouts.split_ranges(_layouts.read(clip_path))
            if not _subs.generate_ass(
                    transcript, clip_start, clip_end, ass_path,
                    split_ranges=split_ranges,
                    max_chars=style["max_chars"], max_duration=style["max_duration"],
                    alignment=style["alignment"], fontsize=style["font_size"],
                    font_name=style["font_name"], font_color=style["font_color"],
                    border_color=style["border_color"], border_width=style["border_width"],
                    highlight_color=style["highlight_color"], effect=style["effect"],
                    base_opacity=style["base_opacity"], uppercase=style["uppercase"]):
                print("   ℹ️ No words in range — clip ships without captions.")
                return None

            _subs.burn_subtitles(
                clip_path, ass_path, out_path,
                alignment=style["alignment"], fontsize=style["font_size"],
                font_name=style["font_name"], font_color=style["font_color"],
                border_color=style["border_color"], border_width=style["border_width"])
            print(f"   💬 Captions burned: {os.path.basename(out_path)}")
            return out_path
        finally:
            if os.path.exists(ass_path):
                try:
                    os.remove(ass_path)
                except OSError:
                    pass
    except Exception as e:
        print(f"   ⚠️ Auto-captions failed ({type(e).__name__}: {e}) — "
              f"delivering the clip without them.")
        return None


def auto_hook_clip(clip_path, clip):
    """Burn the clip's Gemini hook text as a DERIVED file (AUTO_HOOK=1).

    Writes ``hooked_<ts>_<clip filename>`` next to the canonical clip, exactly
    like captions write ``subtitled_<ts>_...``: the canonical stays clean, so
    the hook can later be replaced or removed by walking the prefix back
    (app.py `_strip_burned_hook`). Captions are then burned ON TOP of the
    hooked file, keeping the "captions are always the last layer" invariant.

    Returns (hooked_path, hook_config), or None when skipped or failed — a
    hook problem must never cost the user the clip itself (same fail-open
    contract as auto_caption_clip)."""
    text = (clip.get('viral_hook_text') or '').strip()
    if not text:
        return None
    style = os.environ.get("AUTO_HOOK_STYLE", "classic")
    try:
        seconds = float(os.environ.get("AUTO_HOOK_SECONDS", "5"))
    except ValueError:
        seconds = 5.0
    try:
        from hooks import add_hook_to_video, HOOK_STYLES
        if style not in HOOK_STYLES:
            style = "classic"
        output_dir = os.path.dirname(clip_path)
        out_path = os.path.join(
            output_dir, f"hooked_{int(time.time())}_{os.path.basename(clip_path)}")
        add_hook_to_video(clip_path, text, out_path, position="top",
                          duration=seconds, style=style)
        print(f"   🪝 Hook burned ({style}, {seconds:g}s): {text}")
        return out_path, {"text": text, "style": style, "position": "top",
                          "duration_seconds": seconds}
    except Exception as e:
        print(f"   ⚠️ Auto-hook failed ({type(e).__name__}: {e}) — "
              f"delivering the clip without it.")
        return None


def render_clip(input_video, final_output_video, output_format="auto",
                force_strategy=None, crop_overrides=None):
    """Route a cut clip through the right renderer for the chosen output format.
    vertical/auto -> 9:16 reframe, square -> 1:1 reframe, horizontal -> keep.
    ``force_strategy`` (e.g. 'WIDE'/'TRACK') pins every scene's layout — the
    clip editor's whole-clip framing override. ``crop_overrides`` positions
    individual scenes by hand (the per-scene reframing editor) and wins over
    ``force_strategy`` for the scenes it names."""
    if output_format == "horizontal":
        return finalize_clip_passthrough(input_video, final_output_video)
    aspect = 1.0 if output_format == "square" else ASPECT_RATIO
    return process_video_to_vertical(input_video, final_output_video, aspect_ratio=aspect,
                                     force_strategy=force_strategy,
                                     crop_overrides=crop_overrides)


# Watermark geometry, as fractions of the clip width/height.
#
# Vertical placement is the whole point: the top and bottom strips of a 9:16
# clip are either black bars or blurred filler (GENERAL layout), so a mark up
# there is cropped away without touching a single pixel of real footage. At 40%
# of the height it sits inside the content band — a 16:9 source letterboxed
# into 9:16 spans roughly 34%-66% — so removing the mark means cutting into the
# picture. Left-aligned, like OpusClip's.
WATERMARK_WIDTH_RATIO = 0.30
WATERMARK_MARGIN_RATIO = 0.05
WATERMARK_Y_RATIO = 0.40
WATERMARK_OPACITY = 0.85


def apply_watermark(video_path):
    """Burn the OpenShorts watermark into a finished clip (free plan).

    One re-encode pass on the final file so every output format (TRACK,
    GENERAL, horizontal passthrough) gets the mark, and later subtitle/hook
    re-encodes keep it — they re-encode the already-marked pixels.
    """
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "assets", "watermark.png")
    if not os.path.exists(logo_path):
        print(f"   ⚠️ Watermark asset missing ({logo_path}); clip kept unmarked.")
        return False

    # Scale the lockup from the clip's real width: overlay can't read the other
    # input's size, and computing it here avoids the deprecated scale2ref.
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", video_path],
            stderr=subprocess.STDOUT, timeout=60,
        ).decode().strip().split("x")
        vw, vh = int(probe[0]), int(probe[1])
    except Exception as e:
        print(f"   ⚠️ Could not probe clip for watermark ({e}); clip kept unmarked.")
        return False

    wm_w = max(80, int(vw * WATERMARK_WIDTH_RATIO))
    x = int(vw * WATERMARK_MARGIN_RATIO)
    y = int(vh * WATERMARK_Y_RATIO)
    filt = (
        f"[1:v]scale={wm_w}:-1,format=rgba,"
        f"colorchannelmixer=aa={WATERMARK_OPACITY}[wm];"
        f"[0:v][wm]overlay=x={x}:y={y}"
    )
    tmp_path = video_path + ".wm.mp4"
    cmd = ["ffmpeg", "-y", "-i", video_path, "-i", logo_path,
           "-filter_complex", filt,
           *video_encode_args(QUALITY), "-c:a", "copy", *METADATA_SCRUB,
           "-movflags", "+faststart", tmp_path]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            timeout=1800)
    if result.returncode == 0 and os.path.exists(tmp_path):
        os.replace(tmp_path, video_path)
        return True
    err = (result.stderr or b"").decode(errors="ignore")[-300:]
    print(f"   ⚠️ Watermark pass failed (clip kept unmarked): {err}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False


def process_video_to_vertical(input_video, final_output_video, aspect_ratio=ASPECT_RATIO,
                              force_strategy=None, crop_overrides=None):
    """
    Core logic to reframe a horizontal video to a target aspect ratio using
    scene detection and Active Speaker Tracking (MediaPipe).
    aspect_ratio: width/height of the output (9/16 vertical, 1.0 square).
    force_strategy / crop_overrides pin layouts and scene crops by hand (v2
    engine only — the v1 loop below has no layout concept beyond its own
    classifier).
    """
    # v2 engine: analyze downscaled, render natively in ffmpeg. Any failure
    # falls back to the v1 frame loop below so a v2 edge case can't kill jobs.
    if os.environ.get("REFRAME_ENGINE", "v2").strip().lower() != "v1":
        try:
            import reframe_v2
            t0 = time.time()
            result = reframe_v2.render(input_video, final_output_video, aspect_ratio,
                                       force_strategy=force_strategy,
                                       crop_overrides=crop_overrides)
            print(f"   ⏱️ Reframe v2 total: {time.time() - t0:.1f}s")
            return result
        except Exception as e:
            # Only v2 honours hand-framed scenes and forced layouts. Falling
            # through to v1 would quietly return an automatically framed clip,
            # and the user would see their correction vanish with no reason
            # given — so surface the failure instead of discarding their input.
            if crop_overrides or force_strategy:
                raise RuntimeError(
                    f"manual framing needs the v2 reframe engine, which failed "
                    f"({type(e).__name__}: {e})") from e
            print(f"   ⚠️ Reframe v2 failed ({type(e).__name__}: {e}) — "
                  f"falling back to v1 frame loop")

    # The v1 loop stages its work next to the final file: a silent video track
    # first, then the source audio, muxed together at the end.
    stem = os.path.splitext(final_output_video)[0]
    silent_video_path = stem + ".v1video.mp4"
    audio_track_path = stem + ".v1audio.aac"
    for stale in (silent_video_path, audio_track_path, final_output_video):
        # isfile, not exists: a caller that hands us a directory should not
        # take an EACCES here, and must never have it deleted either.
        if os.path.isfile(stale):
            os.remove(stale)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        # Scene detection found nothing: treat the whole video as one scene.
        print("   ❌ No scenes were detected. Using full video as one scene.")
        probe = cv2.VideoCapture(input_video)
        span = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        probe.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(span, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    original_width, original_height = get_video_resolution(input_video)
    
    # Same delivery floor as the v2 engine — a fallback render is still the clip
    # the user posts, so it must not ship sub-HD. The frame loop below already
    # resizes every cropped frame to these dims, so nothing else changes.
    from reframe_v2 import delivery_size
    OUTPUT_WIDTH, OUTPUT_HEIGHT = delivery_size(original_width, original_height,
                                                aspect_ratio)

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height, aspect_ratio=aspect_ratio)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    # Raw BGR frames stream down a pipe into ffmpeg, which encodes the silent
    # video track; the audio is muxed back in afterwards.
    encoder = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'bgr24',
         '-video_size', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}',
         '-framerate', str(fps), '-i', 'pipe:0',
         *video_encode_args(QUALITY_FAST), '-an', silent_video_path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    reader = cv2.VideoCapture(input_video)
    frame_total = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    # Per-stage wall time (server-side diagnostics; hidden from cloud logs).
    stage_seconds = {'detect': 0.0, 'write': 0.0}
    loop_started = time.time()

    with tqdm(total=frame_total, desc="   Processing", file=sys.stdout) as pbar:
        while reader.isOpened():
            ret, frame = reader.read()
            if not ret:
                break

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1
            
            # Determine Strategy for current frame based on scene
            current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
            
            # Apply Strategy
            if current_strategy == 'GENERAL':
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                
            else:
                # "Single Speaker" -> Track & Crop

                # Detect every Nth frame for performance (cameraman smooths in
                # between); the much heavier YOLO fallback gets its own stride.
                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                if is_scene_start and SCENE_CUT_RESET:
                    speaker_tracker.reset()
                    cameraman.begin_scene()

                # Always detect on a cut, whatever the stride: the new shot's
                # subject has to be found before the first frame is framed.
                if frame_number % DETECT_STRIDE == 0 or (is_scene_start and SCENE_CUT_RESET):
                    t_det = time.time()
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    elif frame_number % YOLO_FALLBACK_STRIDE == 0 or (is_scene_start and SCENE_CUT_RESET):
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)
                    stage_seconds['detect'] += time.time() - t_det

                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)

                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)

            t_wr = time.time()
            encoder.stdin.write(output_frame.tobytes())
            stage_seconds['write'] += time.time() - t_wr
            frame_number += 1
            pbar.update(1)
    
    loop_total = time.time() - loop_started
    other = loop_total - stage_seconds['detect'] - stage_seconds['write']
    print(f"\n   ⏱️ Frame loop: {loop_total:.1f}s total — "
          f"detect {stage_seconds['detect']:.1f}s, "
          f"encode-wait {stage_seconds['write']:.1f}s, "
          f"decode+render {other:.1f}s ({frame_number} frames)")

    encoder.stdin.close()
    encode_log = encoder.stderr.read().decode()
    encoder.wait()
    reader.release()

    if encoder.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", encode_log)
        return False

    print("\n   🔊 Step 5: Extracting audio...")
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_video, '-vn', '-c:a', 'copy', audio_track_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")

    print("\n   ✨ Step 6: Merging...")
    mux = ['ffmpeg', '-y', '-i', silent_video_path]
    if os.path.exists(audio_track_path):
        mux += ['-i', audio_track_path]
    mux += ['-c', 'copy', *METADATA_SCRUB, '-movflags', '+faststart', final_output_video]
    try:
        subprocess.run(mux, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False

    for leftover in (silent_video_path, audio_track_path):
        if os.path.exists(leftover):
            os.remove(leftover)

    return True

# --- Transcript checkpoint (survive a redeploy without paying twice) ---------
# A job interrupted by a container restart is re-run from its resume manifest
# (app.py) with the SAME output directory. Transcription is the slow, paid part
# of the pipeline that ran before the interruption, so the finished transcript
# is left in the job directory and picked up by the re-run instead of
# transcribing again. Same shape and same validation as --transcript.
TRANSCRIPT_CHECKPOINT = ".transcript_checkpoint.json"


def _checkpoint_source_key(input_video, duration):
    """What ties a checkpoint to ONE source. Not the path: a resumed cloud job
    re-downloads to the same name, and the CLI may be pointed at a different
    file in a directory where an earlier run died. Name plus duration is what
    both the server and a careful CLI user keep stable across the two runs."""
    return {"name": os.path.basename(input_video), "duration": round(float(duration), 1)}


def save_transcript_checkpoint(output_dir, transcript, input_video, duration):
    """Best effort: a failure here must never fail the job."""
    try:
        payload = {"source": _checkpoint_source_key(input_video, duration),
                   "transcript": transcript}
        with open(os.path.join(output_dir, TRANSCRIPT_CHECKPOINT), "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as e:
        print(f"⚠️ Could not save transcript checkpoint: {e}")


def load_transcript_checkpoint(output_dir, input_video, duration):
    """The transcript an interrupted run left behind for THIS source, or None.

    A checkpoint for a different source (a CLI run that died, then a new video
    processed in the same directory) is ignored, not reused."""
    path = os.path.join(output_dir, TRANSCRIPT_CHECKPOINT)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        source = payload.get("source") or {}
        expected = _checkpoint_source_key(input_video, duration)
        if source.get("name") != expected["name"] \
                or abs(float(source.get("duration", -1)) - expected["duration"]) > 0.5:
            print("⏭️ Transcript checkpoint belongs to another source — ignoring it.")
            return None
        transcript = payload.get("transcript") or {}
        if not transcript.get("segments"):
            raise ValueError("checkpoint has no segments")
        return transcript
    except Exception as e:
        print(f"⚠️ Ignoring unusable transcript checkpoint ({e}).")
        return None


def clear_transcript_checkpoint(output_dir):
    try:
        os.remove(os.path.join(output_dir, TRANSCRIPT_CHECKPOINT))
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ Could not remove transcript checkpoint: {e}")


def transcribe_video(video_path):
    print("🎙️  Transcribing video...")
    from transcribe_backends import transcribe_media

    transcript = transcribe_media(video_path)

    print(f"   Detected language '{transcript['language']}', "
          f"{len(transcript['segments'])} segments")
    for segment in transcript['segments']:
        # Print progress to keep user informed (and prevent timeouts feeling)
        print(f"   [{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    return transcript

def _run_gemini_stage(client, model_name, prompt, schema):
    """One schema-enforced model call with transient-error backoff.
    Returns (parsed_dict, cost_analysis).

    With an OpenAI-compatible server configured (``llm_backend.active()``)
    the call goes there instead of Gemini and ``client`` is unused; the
    retry policy is shared because a local server has the same failure
    shapes (connection refused while the model loads, a truncated body,
    a 5xx from a busy vLLM)."""
    use_local = llm_backend.active()
    config = None if use_local else genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
    )
    key_pool = api_key_rotator.get_all_keys("gemini") if not use_local else []
    max_attempts = max(3, len(key_pool))
    for attempt in range(1, max_attempts + 1):
        try:
            if use_local:
                return llm_backend.generate_json(prompt, schema, model=model_name)
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            # Policy blocks are deterministic — retrying only burns quota and
            # time, and the user deserves the real reason instead of a generic
            # "empty response" (prod 23-jul: PROHIBITED_CONTENT on every try).
            gemini_worker.raise_if_blocked(response)
            # Parsing lives inside the retry loop on purpose: Gemini sometimes
            # returns 200 with an empty body, which raises here rather than at
            # the call. Retrying that recovered every occurrence seen in prod
            # (22-jul-2026) — the same payload succeeds on the next attempt.
            parsed_obj = getattr(response, "parsed", None)
            if parsed_obj is not None:
                parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
            else:
                parsed = gemini_worker._parse_json_response_text(
                    gemini_worker._get_response_text(response))
            if not use_local:
                api_key_rotator.report_gemini_success(api_key_rotator.get_gemini_key())
            return parsed, gemini_worker._calculate_cost_analysis(response, model_name)
        except gemini_worker.GeminiBlockedError:
            raise  # deterministic policy block — never retry
        except Exception as e:
            msg = str(e)
            rotatable = (not use_local) and api_key_rotator.is_rotatable_error(e)
            transient = any(tok in msg for tok in (
                '503', 'UNAVAILABLE', '429', 'RESOURCE_EXHAUSTED',
                '500', 'INTERNAL', 'overloaded', 'Deadline',
                'empty response body', 'did not contain a JSON object',
                'Failed to parse Gemini JSON response',
                # OpenAI-compatible servers: model still loading, busy, or a
                # small model that skipped a required field this time.
                'ConnectError', 'ReadTimeout', 'RemoteProtocolError', '502', '504',
                'validation error')) or rotatable
            if attempt == max_attempts or not transient:
                raise
            if rotatable:
                cur_k = api_key_rotator.get_gemini_key()
                api_key_rotator.report_gemini_failure(cur_k, error=e)
                new_k = api_key_rotator.get_gemini_key()
                if new_k:
                    print(f"🔄 Rotated Gemini API key to {api_key_rotator.mask_key(new_k)} due to quota/rate limit.")
                    try:
                        from google import genai
                        client = genai.Client(api_key=new_k)
                    except Exception:
                        pass
                    if new_k != cur_k:
                        time.sleep(0.5)
                        continue
            wait = 5 * (2 ** (attempt - 1))
            who = "LLM server" if use_local else "Gemini"
            print(f"⚠️ {who} transient error (attempt {attempt}/{max_attempts}), retrying in {wait}s: {msg[:150]}")
            time.sleep(wait)


def _run_stage_split(client, model_name, items, build_prompt, schema, key, costs, label):
    """Run a Gemini stage over ``items``; on a policy block, bisect.

    Google's prompt filter (PROHIBITED_CONTENT) fires on some COMBINATIONS of
    transcript windows that pass individually (27-aug-2026: windows 5+6 of a
    software walkthrough blocked 3/3, each alone fine, all three models).
    A block is deterministic for a given prompt, so instead of failing the
    job the batch is split in halves until the offending combination is
    isolated; a single item that still blocks is dropped with a log line.
    Returns the merged list found under ``key`` in each response."""
    if not items:
        return []
    prompt = build_prompt(items)
    try:
        parsed, cost = _run_gemini_stage(client, model_name, prompt, schema)
        if cost:
            costs.append(cost)
        return list(parsed.get(key) or [])
    except gemini_worker.GeminiBlockedError as e:
        if len(items) == 1:
            print(f"   🚫 {label}: Gemini blocked window {items[0].get('id')} on its own; skipping it ({e})")
            return []
        mid = len(items) // 2
        print(f"   🚫 {label}: Gemini blocked a batch of {len(items)}; retrying as {mid} + {len(items) - mid}")
        return (_run_stage_split(client, model_name, items[:mid], build_prompt, schema, key, costs, label)
                + _run_stage_split(client, model_name, items[mid:], build_prompt, schema, key, costs, label))


def score_batch_size():
    """Transcript windows per scoring call: ``LLM_SCORE_BATCH`` if set, else
    8 for Gemini (1M context) and 3 for an OpenAI-compatible server."""
    raw = os.environ.get("LLM_SCORE_BATCH", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 3 if llm_backend.active() else 8


def _heuristic_transcript_clips(transcript_result, video_duration, min_secs=15, max_secs=60, target_clips=3):
    """Heuristic clip generation when no LLM API key is configured."""
    try:
        import viral_brain
    except Exception:
        viral_brain = None

    segments = transcript_result.get("segments", []) if transcript_result else []
    if not segments:
        return []

    min_b, max_b = clip_duration_bounds()
    min_secs = max(5, int(min_b))
    max_secs = max(min_secs + 5, int(max_b))

    candidates = []
    for i, seg in enumerate(segments):
        start_t = float(seg.get("start", 0.0))
        text_parts = [seg.get("text", "")]
        end_t = float(seg.get("end", 0.0))

        for next_seg in segments[i+1:]:
            dur = float(next_seg.get("end", 0.0)) - start_t
            if dur > max_secs:
                break
            text_parts.append(next_seg.get("text", ""))
            end_t = float(next_seg.get("end", 0.0))
            if dur >= min_secs:
                full_text = " ".join(text_parts).strip()
                if viral_brain:
                    assessment = viral_brain.evaluate_viral_clip(full_text, dur)
                    candidates.append({
                        "start": round(start_t, 2),
                        "end": round(end_t, 2),
                        "predicted_score": assessment.overall_score,
                        "video_description_for_tiktok": assessment.tiktok_description,
                        "video_description_for_instagram": assessment.instagram_description,
                        "video_title_for_youtube_short": assessment.suggested_title,
                        "viral_hook_text": assessment.viral_hook_text,
                    })
                else:
                    candidates.append({
                        "start": round(start_t, 2),
                        "end": round(end_t, 2),
                        "predicted_score": 85,
                        "video_description_for_tiktok": full_text[:100],
                        "video_description_for_instagram": full_text[:100],
                        "video_title_for_youtube_short": full_text[:50],
                        "viral_hook_text": full_text.split(".")[0][:30],
                    })

    candidates.sort(key=lambda c: c["predicted_score"], reverse=True)
    chosen = []
    for c in candidates:
        overlap = False
        for ch in chosen:
            if not (c["end"] <= ch["start"] or c["start"] >= ch["end"]):
                overlap = True
                break
        if not overlap:
            chosen.append(c)
            if len(chosen) >= target_clips:
                break

    if not chosen and segments:
        dur = max(1.0, float(segments[-1].get("end", 0.0)) - float(segments[0].get("start", 0.0)))
        full_text = " ".join(s.get("text", "") for s in segments)
        hook = "WATCH THIS! 🤯"
        title = full_text[:50] or "Highlight Moment"
        if viral_brain:
            assessment = viral_brain.evaluate_viral_clip(full_text, dur)
            hook = assessment.viral_hook_text
            title = assessment.suggested_title
        chosen.append({
            "start": round(float(segments[0].get("start", 0.0)), 2),
            "end": round(float(segments[-1].get("end", 0.0)), 2),
            "predicted_score": 88,
            "video_description_for_tiktok": full_text[:100],
            "video_description_for_instagram": full_text[:100],
            "video_title_for_youtube_short": title,
            "viral_hook_text": hook,
        })
    return chosen


def _heuristic_visual_clips(video_path, video_duration, target_clips=3):
    """Heuristic visual highlight detector for silent videos when no vision AI key is present."""
    shorts = []
    try:
        from highlight_detector import detect_highlights
        highlights = detect_highlights(video_path, target_clips=target_clips)
        for i, h in enumerate(highlights):
            shorts.append({
                "start": round(h.start, 2),
                "end": round(h.end, 2),
                "predicted_score": int(min(99, max(60, h.score * 100))),
                "video_description_for_tiktok": f"Epic highlight moment #{i+1} 🔥",
                "video_description_for_instagram": f"Unreal peak moment #{i+1} ⚡",
                "video_title_for_youtube_short": f"Best Moment #{i+1} That You Won't Believe",
                "viral_hook_text": "WAIT FOR IT... 😱",
            })
    except Exception as e:
        print(f"⚠️ Highlight detector notice ({e}) — falling back to windowing.")

    if not shorts:
        min_b, max_b = clip_duration_bounds()
        clip_dur = min(float(max_b), max(float(min_b), video_duration))
        shorts.append({
            "start": 0.0,
            "end": round(clip_dur, 2),
            "predicted_score": 85,
            "video_description_for_tiktok": "Best moment from this video 🔥",
            "video_description_for_instagram": "Must watch highlight ⚡",
            "video_title_for_youtube_short": "You Won't Believe What Happened",
            "viral_hook_text": "WATCH THIS! 🤯",
        })
    return shorts


def get_viral_clips(transcript_result, video_duration):
    """Two-pass clip selection: score transcript windows, then detail the best.

    Windowing gives even coverage on long videos (a single call over the whole
    transcript clusters picks near the start), and the cheap scoring pass keeps
    the expensive detail reasoning focused on the shortlist. Cuts are snapped to
    word boundaries so clips don't start/end mid-word.
    """
    language = str(transcript_result.get('language') or 'unknown')
    is_local_mode = llm_backend.active() or llm_backend.is_colab() or os.environ.get("FORCE_LOCAL_ONLY") == "1"
    if is_local_mode:
        # Self-hosted / Colab local text model: no Google key needed for this stage.
        client = None
        model_name = llm_backend.model_name()
        endpoint = llm_backend.base_url() or "http://127.0.0.1:11434/v1"
        print(f"\U0001f916  Analyzing with local LLM ({model_name}) at {endpoint} (2-pass: score → detail)...")
    else:
        print("\U0001f916  Analyzing with Gemini (2-pass: score → detail)...")
        raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
        if raw_keys:
            api_key_rotator.register_keys("gemini", raw_keys)
        api_key = api_key_rotator.get_gemini_key()
        if not api_key:
            print("ℹ️ No GEMINI_API_KEY detected — running heuristic Virality Engine (viral_brain)...")
            return {
                "shorts": _heuristic_transcript_clips(transcript_result, video_duration),
                "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
            }
        client = genai.Client(api_key=api_key)
        model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
    print(f"\U0001f916  Model: {model_name} | language: {language}")

    # Full word list — ground truth for snapping cut points.
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({'w': word['word'], 's': word['start'], 'e': word['end']})

    try:
        # Scoring windows must be able to CONTAIN a max-length clip (the detail
        # prompt keeps clips inside their candidate window), so scale them with
        # the requested band — a user asking for 60-90s clips on the default
        # 90s windows would get clips squeezed against the window walls.
        min_secs, max_secs = clip_duration_bounds()
        windows = build_transcript_windows(
            transcript_result, video_duration,
            window_seconds=max(90, int(max_secs * 1.5)))
        print(f"   Built {len(windows)} scoring window(s).")
        costs = []

        # --- Pass 1: score windows in batches, keep the highest-scoring ---
        scored = []
        # Local models usually run with a 4-8k context (Ollama defaults to
        # 4096 unless OLLAMA_CONTEXT_LENGTH says otherwise) and 8 windows of
        # transcript do not fit; a silently truncated prompt scores garbage.
        SCORE_BATCH = score_batch_size()
        def _payload(ws):
            return [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]} for w in ws]

        def _score_prompt(ws):
            return gemini_worker.SCORE_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                windows_json=json.dumps(_payload(ws), ensure_ascii=False))

        for b in range(0, len(windows), SCORE_BATCH):
            scored.extend(_run_stage_split(
                client, model_name, windows[b:b + SCORE_BATCH], _score_prompt,
                gemini_worker.ScoreResponse, "windows", costs, "score"))

        # Shortlist the top windows; scale with duration so long videos surface
        # more candidates without exploding the detail call.
        scored.sort(key=lambda w: w.get("score", 0), reverse=True)
        target = max(3, min(10, int(video_duration // 90) + 2))
        by_id = {w["id"]: w for w in windows}
        shortlist = [by_id[w["id"]] for w in scored[:target] if w.get("id") in by_id]
        if not shortlist:
            shortlist = windows[:target]  # scoring returned nothing usable
        print(f"   Shortlisted {len(shortlist)} window(s) for detail.")

        # --- Pass 2: detailed clip extraction on the shortlist ---
        min_clips, max_clips = clip_count_targets(len(shortlist))

        def _detail_prompt(ws):
            # A split batch keeps the full clip-count band: a short list can
            # still hold the best clips, and the model returns fewer anyway.
            return gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                min_clips=min_clips, max_clips=max_clips,
                min_secs=min_secs, max_secs=max_secs,
                windows_json=json.dumps(_payload(ws), ensure_ascii=False))

        shorts = _run_stage_split(client, model_name, shortlist, _detail_prompt,
                                  gemini_worker.DetailResponse, "shorts", costs, "detail")
        if len(shorts) > max_clips:
            # By score, never by position: the results arrive in transcript
            # order, so slicing kept the earliest clips and silently dropped
            # the back half of the video. See trim_to_best.
            dropped = len(shorts) - max_clips
            shorts = trim_to_best(shorts, max_clips)
            print(f"   Kept the {max_clips} best-scoring clip(s) of "
                  f"{max_clips + dropped}.")
        # Snap each proposed clip onto real word boundaries (+ a bit of silence).
        for s in shorts:
            ns, ne = snap_clip_to_words(s.get("start", 0), s.get("end", 0), words, video_duration,
                                        min_duration=min_secs, max_duration=max_secs)
            s["start"], s["end"] = ns, ne

        # Aggregate cost across both passes.
        cost_analysis = None
        if costs:
            cost_analysis = {
                "input_tokens": sum(c.get("input_tokens", 0) for c in costs),
                "output_tokens": sum(c.get("output_tokens", 0) for c in costs),
                "total_cost": sum(c.get("total_cost", 0) for c in costs),
                "model": model_name,
            }
            print(f"\U0001f4b0 Total cost ({model_name}, 2-pass, {len(costs)} calls): ${cost_analysis['total_cost']:.6f}")

        if not shorts:
            print("⚠️ 2-pass returned no clips — activating heuristic virality engine...")
            return {
                "shorts": _heuristic_transcript_clips(transcript_result, video_duration),
                "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
            }

        result = {"shorts": shorts}
        if cost_analysis:
            result["cost_analysis"] = cost_analysis
        return result
    except gemini_worker.GeminiBlockedError as e:
        # Content-policy rejection: propagate so the job fails with the real
        # reason instead of a generic "no clips found".
        print(f"🚫 {e}")
        raise
    except Exception as e:
        who = "Local LLM" if (llm_backend.active() or llm_backend.is_colab()) else "Gemini"
        print(f"❌ {who} Error: {e} — activating heuristic virality engine fallback...")
        return {
            "shorts": _heuristic_transcript_clips(transcript_result, video_duration),
            "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
        }


# --- Speech too sparse to clip by transcript -------------------------------
# The vision path used to fire only on a missing audio TRACK. A nursery-rhyme
# video or a dashcam drive has audio, so it went through transcription, came
# back as one segment ("Uh uh"), produced one scoring window and Gemini
# returned no clips — three failed jobs on 25-aug-2026, one user twice. Speech
# is ~120-160 words/min; below these floors there is nothing to clip by words.
MIN_SPEECH_WORDS_PER_MIN = float(os.environ.get("MIN_SPEECH_WORDS_PER_MIN", "5"))
MIN_SPEECH_WORDS = int(os.environ.get("MIN_SPEECH_WORDS", "8"))


def speech_is_sparse(transcript, duration):
    """True when the transcript is too thin to drive clip selection."""
    words = sum(len((seg.get("text") or "").split())
                for seg in (transcript or {}).get("segments", []))
    minutes = max(float(duration or 0) / 60.0, 1e-6)
    return words < MIN_SPEECH_WORDS or words / minutes < MIN_SPEECH_WORDS_PER_MIN


def get_visual_clips(video_path, video_duration, language="en"):
    """Clip a SILENT video by vision: Gemini watches the footage and picks the
    most engaging visual moments (no transcript). Returns the same
    {"shorts", "cost_analysis"} shape as get_viral_clips, or None."""
    if llm_backend.is_colab() or os.environ.get("FORCE_LOCAL_ONLY") == "1":
        print("🎥  Colab zero-API mode active — running heuristic Highlight Detector (no external APIs)...")
        return {
            "shorts": _heuristic_visual_clips(video_path, video_duration),
            "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
        }
    print("🎥  Silent video — analyzing with Gemini vision (no transcript)...")
    raw_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
    if raw_keys:
        api_key_rotator.register_keys("gemini", raw_keys)
    api_key = api_key_rotator.get_gemini_key()
    if not api_key:
        print("ℹ️ No GEMINI_API_KEY detected — running heuristic Highlight Detector...")
        return {
            "shorts": _heuristic_visual_clips(video_path, video_duration),
            "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
        }
    key_pool = api_key_rotator.rotator.get_keys("gemini")
    max_attempts = max(2, len(key_pool))
    file_upload = None

    for attempt in range(1, max_attempts + 1):
        try:
            api_key = api_key_rotator.get_gemini_key()
            if not api_key:
                break
            client = genai.Client(api_key=api_key)
            model_name = os.environ.get("GEMINI_MODEL") or 'gemini-3.1-flash-lite'
            print(f"🎥  Model: {model_name} (key {api_key_rotator.mask_key(api_key)}) | uploading {os.path.basename(video_path)}…")

            file_upload = client.files.upload(file=video_path)
            deadline = time.time() + 180
            while True:
                info = client.files.get(name=file_upload.name)
                state = str(getattr(getattr(info, "state", info), "name", "")).upper()
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise RuntimeError("Gemini file processing reported FAILED state.")
                if time.time() > deadline:
                    raise TimeoutError("Gemini video processing timed out.")
                time.sleep(2)

            def _env_int(name, default):
                try:
                    return max(1, int(os.environ.get(name, "")))
                except ValueError:
                    return default

            v_min_clips = _env_int("CLIP_TARGET_MIN", 3)
            v_max_clips = max(v_min_clips, _env_int("CLIP_TARGET_MAX", 15))
            v_min_secs, v_max_secs = clip_duration_bounds()
            prompt = gemini_worker.VISUAL_PROMPT_TEMPLATE.format(
                video_duration=video_duration, language=language,
                min_clips=v_min_clips, max_clips=v_max_clips,
                min_secs=v_min_secs, max_secs=v_max_secs)
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=gemini_worker.VisualResponse,
            )
            response = client.models.generate_content(
                model=model_name, contents=[file_upload, prompt], config=config)
            gemini_worker.raise_if_blocked(response)
            parsed = json.loads(response.text)
            shorts = parsed.get("shorts") or []
            # Clamp to the real duration; drop anything degenerate.
            clean = []
            for s in shorts:
                s["start"] = max(0.0, float(s.get("start", 0)))
                s["end"] = min(float(video_duration), float(s.get("end", 0)))
                if s["end"] - s["start"] >= 1.0:
                    clean.append(s)
            if not clean:
                print("⚠️ Vision pass returned no usable clips — activating heuristic highlight detector...")
                return {
                    "shorts": _heuristic_visual_clips(video_path, video_duration),
                    "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
                }

            api_key_rotator.report_gemini_success(api_key)
            cost = gemini_worker._calculate_cost_analysis(response, model_name)
            if cost:
                print(f"💰 Vision cost ({model_name}): ${cost.get('total_cost', 0):.6f}")
            result = {"shorts": clean}
            if cost:
                result["cost_analysis"] = cost
            return result
        except gemini_worker.GeminiBlockedError as e:
            print(f"🚫 {e}")
            raise
        except Exception as e:
            rotatable = api_key_rotator.is_rotatable_error(e)
            if rotatable and api_key:
                api_key_rotator.report_gemini_failure(api_key, error=e)
                new_k = api_key_rotator.get_gemini_key()
                if new_k and new_k != api_key:
                    print(f"🔄 Rotated vision Gemini API key to {api_key_rotator.mask_key(new_k)} due to error.")
                    time.sleep(0.5)
                    continue
            print(f"⚠️ Gemini vision attempt {attempt}/{max_attempts} error: {e}")
            if attempt == max_attempts:
                break
            time.sleep(2)
        finally:
            if file_upload is not None:
                try:
                    client.files.delete(name=file_upload.name)
                except Exception:
                    pass
                file_upload = None

    print("ℹ️ Falling back to heuristic Highlight Detector for visual clips...")
    return {
        "shorts": _heuristic_visual_clips(video_path, video_duration),
        "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoCrop-Vertical with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--format', type=str, default="auto", choices=["auto", "vertical", "horizontal", "square"],
                        help="Output aspect: vertical/auto (9:16), horizontal (keep 16:9), square (1:1).")
    parser.add_argument('--transcript', type=str,
                        help="Path to a precomputed transcript JSON (transcribe_media shape); skips transcription.")
    parser.add_argument('--clip-min-seconds', type=float, default=None,
                        help="Minimum duration for extracted clips in seconds.")
    parser.add_argument('--clip-max-seconds', type=float, default=None,
                        help="Maximum duration for extracted clips in seconds.")
    parser.add_argument('--target-clips', '--max-clips', type=int, default=None, dest="target_clips",
                        help="Target number of clips to extract.")
    parser.add_argument('--live-record-seconds', type=int, default=None,
                        help="Duration in seconds to capture if URL is an active YouTube Live stream (default: 900s).")
    return parser


def process_video_pipeline(
    input_video_or_url: str,
    output_dir: Optional[str] = None,
    is_url: bool = True,
    output_format: str = "vertical",
    target_clips: Optional[int] = None,
    clip_min_seconds: Optional[float] = None,
    clip_max_seconds: Optional[float] = None,
    live_record_seconds: Optional[int] = None,
    keep_original: bool = False,
    transcript_path: Optional[str] = None,
    skip_analysis: bool = False,
) -> dict:
    """End-to-end programmatic entrypoint for processing a video or YouTube URL.

    Extracts viral short clips, reframes to vertical 9:16 (or chosen format),
    applies AI face-tracking and burned subtitles without requiring CLI interaction.
    """
    if live_record_seconds:
        os.environ["LIVE_RECORD_SECONDS"] = str(live_record_seconds)
    if clip_min_seconds:
        os.environ["CLIP_MIN_SECONDS"] = str(clip_min_seconds)
    if clip_max_seconds:
        os.environ["CLIP_MAX_SECONDS"] = str(clip_max_seconds)
    if target_clips:
        os.environ["CLIP_TARGET_MIN"] = str(target_clips)
        os.environ["CLIP_TARGET_MAX"] = str(target_clips)

    script_start_time = time.time()

    def _ensure_dir(path: str) -> str:
        if path:
            os.makedirs(path, exist_ok=True)
        return path

    if is_url:
        from security_utils import normalize_media_url
        url = normalize_media_url(input_video_or_url)
        resolved_output_dir = _ensure_dir(output_dir or "./output")
        input_video, video_title = download_youtube_video(url, resolved_output_dir)
    else:
        input_video = input_video_or_url
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        if output_dir and not skip_analysis:
            resolved_output_dir = _ensure_dir(output_dir)
        elif output_dir and os.path.isdir(output_dir):
            resolved_output_dir = output_dir
        elif output_dir and not os.path.isdir(output_dir):
            resolved_output_dir = os.path.dirname(output_dir) or "."
        else:
            resolved_output_dir = os.path.dirname(input_video) or "."

    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")

    # Layout choice is per SOURCE video, not per clip
    if layout_picker.ENABLED:
        try:
            _cap = cv2.VideoCapture(input_video)
            _fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
            _duration = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) / _fps
            _w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            _h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _cap.release()
            from reframe_v2 import source_already_fits
            if _w and _h and source_already_fits(_w, _h, ASPECT_RATIO):
                print(f"   ↕️  Source is {_w}x{_h} — already vertical, no layout to pick.")
            else:
                layout_picker.pick_and_apply(input_video, _duration)
        except Exception as e:
            print(f"⚠️ Layout choice skipped ({e}) — using the default layout.")

    ready_clips = []
    clips_data = None
    duration = 0.0

    if skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = output_dir
        if (not output_file or os.path.isdir(output_file)
                or output_file.endswith(("/", os.sep))
                or not os.path.splitext(output_file)[1]):
            output_file = os.path.join(output_file or resolved_output_dir, f"{video_title}_vertical.mp4")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        render_clip(input_video, output_file, output_format)
        ready_clips.append(output_file)
    else:
        # Get duration
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = frame_count / fps if fps > 0 else 0
        print(f"🎥 Video Duration: {duration:.2f}s ({frame_count} frames @ {fps:.2f} fps)")

        # 3. Transcribe
        from transcribe_backends import NoAudioError
        transcript = None
        if transcript_path:
            try:
                with open(transcript_path, 'r', encoding="utf-8") as f:
                    transcript = json.load(f)
                if not transcript.get('segments'):
                    raise ValueError("transcript has no segments")
                print(f"⏩ Reusing precomputed transcript ({len(transcript['segments'])} segments).")
            except Exception as e:
                print(f"⚠️ Could not use precomputed transcript ({e}) — transcribing normally.")
                transcript = None

        if transcript is None:
            transcript = load_transcript_checkpoint(resolved_output_dir, input_video, duration)
            if transcript is not None:
                print(f"♻️ Reusing transcript checkpoint ({len(transcript['segments'])} segments).")

        if transcript is None:
            try:
                transcript = transcribe_video(input_video)
                save_transcript_checkpoint(resolved_output_dir, transcript, input_video, duration)
            except (NoAudioError, Exception) as e:
                print(f"🔇 Transcription notice ({e}) — switching to visual highlight analysis.")
                transcript = None

        if transcript is not None and speech_is_sparse(transcript, duration):
            n_words = sum(len((sg.get("text") or "").split()) for sg in transcript["segments"])
            print(f"🔇 Only {n_words} word(s) of speech in {duration:.0f}s — switching to visual analysis.")
            transcript = None

        # 4. Clip Detection (Local model or heuristic)
        if transcript is not None:
            clips_data = get_viral_clips(transcript, duration)
        else:
            clips_data = get_visual_clips(input_video, duration)

        if not clips_data or 'shorts' not in clips_data or not clips_data.get('shorts'):
            print("⚠️ AI clip detection returned empty — generating heuristic highlight clips...")
            if transcript is not None:
                clips_data = {
                    "shorts": _heuristic_transcript_clips(transcript, duration),
                    "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
                }
            else:
                clips_data = {
                    "shorts": _heuristic_visual_clips(input_video, duration),
                    "cost_analysis": {"total_cost": 0.0, "currency": "USD", "provider": "heuristic"}
                }

        if not clips_data or 'shorts' not in clips_data or not clips_data.get('shorts'):
            raise RuntimeError("Clip detection failed — unable to identify usable clips for this video.")

        print(f"🔥 Found {len(clips_data['shorts'])} clips!")

        clips_data['transcript'] = transcript or {"language": "none", "segments": []}
        clips_data['source_video'] = os.path.basename(input_video)
        clips_data['output_format'] = output_format
        metadata_file = os.path.join(resolved_output_dir, f"{video_title}_metadata.json")
        with open(metadata_file, 'w', encoding="utf-8") as f:
            json.dump(clips_data, f, indent=2)

        # 5. Process clips in parallel
        def _process_one_clip(i, clip):
            start = clip['start']
            end = clip['end']
            print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
            print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")

            clip_filename = f"{video_title}_clip_{i+1}.mp4"
            clip_temp_path = os.path.join(resolved_output_dir, f"temp_{clip_filename}")
            clip_final_path = os.path.join(resolved_output_dir, clip_filename)

            try:
                cut_command = [
                    'ffmpeg', '-y',
                    '-ss', str(start),
                    '-to', str(end),
                    '-i', input_video,
                    *video_encode_args(QUALITY_FAST),
                    *audio_encode_args(),
                    clip_temp_path
                ]
                proc = subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if proc.returncode != 0 or not os.path.exists(clip_temp_path) or os.path.getsize(clip_temp_path) == 0:
                    err_msg = proc.stderr.decode("utf-8", errors="ignore") if proc.stderr else "unknown error"
                    print(f"❌ Failed to cut clip {i+1} with FFmpeg: {err_msg[:200]}")
                    return False

                success = render_clip(clip_temp_path, clip_final_path, output_format)
                if success and os.environ.get("WATERMARK") == "1":
                    apply_watermark(clip_final_path)
                deliver_path = clip_final_path
                import layout_ranges as _layouts
                clip['layout_ranges'] = _layouts.read(clip_final_path)
                if success and hook_grounding.wanted(clip['layout_ranges'], end - start):
                    hook_grounding.reground(clip_final_path, clip, transcript, start, end)
                if success and os.environ.get("AUTO_HOOK") == "1":
                    hooked = auto_hook_clip(clip_final_path, clip)
                    if hooked:
                        deliver_path, clip['auto_hook'] = hooked
                if success:
                    captioned = auto_caption_clip(
                        deliver_path, transcript, start, end,
                        split_ranges=_layouts.split_ranges(clip['layout_ranges']))
                    final_ready = captioned or deliver_path
                    ready_clips.append(final_ready)
                    print(f"   ✅ Clip {i+1} ready: {final_ready}")
                    print(f"CLIP_READY {i} "
                          f"{os.path.basename(captioned or deliver_path)}")
                return success
            finally:
                if os.path.exists(clip_temp_path):
                    os.remove(clip_temp_path)

        clip_workers = max(int(os.environ.get("CLIP_WORKERS", "4")), 1)
        shorts = clips_data['shorts']
        with ThreadPoolExecutor(max_workers=min(clip_workers, len(shorts))) as pool:
            futures = {pool.submit(_process_one_clip, i, clip): i for i, clip in enumerate(shorts)}
            for future in as_completed(futures):
                i = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"   ❌ Clip {i+1} failed: {type(e).__name__}: {e}")

        if any('auto_hook' in c or 'hook_grounding' in c for c in shorts):
            with open(metadata_file, 'w', encoding="utf-8") as f:
                json.dump(clips_data, f, indent=2)

    # Clean up original if requested
    if is_url and not keep_original and os.path.exists(input_video):
        try:
            os.remove(input_video)
            print("🗑️  Cleaned up downloaded video.")
        except Exception:
            pass

    if not skip_analysis:
        clear_transcript_checkpoint(resolved_output_dir)

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
    return {
        "success": True,
        "video_title": video_title,
        "input_video": input_video,
        "output_dir": resolved_output_dir,
        "clips": ready_clips,
        "metadata": clips_data,
        "duration": duration,
        "execution_time": total_time,
    }


if __name__ == '__main__':
    parser = build_arg_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"ℹ️ Note: Ignoring unrecognized CLI arguments: {unknown}")

    is_url = bool(args.url)
    input_target = args.url if is_url else args.input

    process_video_pipeline(
        input_video_or_url=input_target,
        output_dir=args.output,
        is_url=is_url,
        output_format=args.format,
        target_clips=args.target_clips,
        clip_min_seconds=args.clip_min_seconds,
        clip_max_seconds=args.clip_max_seconds,
        live_record_seconds=args.live_record_seconds,
        keep_original=args.keep_original,
        transcript_path=args.transcript,
        skip_analysis=args.skip_analysis,
    )
