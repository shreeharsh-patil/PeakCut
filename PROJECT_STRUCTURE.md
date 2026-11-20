# OpenShorts & PeakCut — Project Structure & Architecture

This document provides a clean overview of the repository's file structure, subsystem organization, and key entry points.

---

## 📁 Directory Layout

```text
D:\yt
├── api/                    # Vercel Serverless Functions (cron, index, status, trigger)
├── alembic/                # SQLAlchemy database migration scripts
├── anti_reused_pipeline/   # Content uniqueness & transformative fair-use pipeline
├── assets/                 # Branding assets, watermarks, and audio beds
├── cli/                    # Standalone OpenShorts CLI package
├── clipmind/               # Autonomous trend-to-shorts AI pipeline engine
├── cloud/                  # Cloud SaaS backend (Auth, Billing, Metering, Models)
├── dashboard/              # React + Vite frontend web application
├── docs/                   # Architectural guides, DMCA, and compliance policies
├── engine/                 # Core Python video processing engine (modular packages)
│   ├── layouts/            # Active speaker tracking, 9:16 vertical reframing, multi-panel & split layouts
│   ├── editing/            # Video filtergraphs, silence cutting, subtitles, hook text, color grading
│   ├── intelligence/       # Multimodal Gemini analysis, transcription, translation, viral scoring
│   ├── discovery/          # Stream ingestion, rolling clipper, channel watcher, scene detection
│   ├── services/           # Publishing queue, API key rotator, alerts, storage, and telemetry
│   ├── studio/             # PeakCut studio bridge, supercharged routers, and FastMCP server
│   └── utils/              # Batch processing, quality probe, and security utilities
├── examples/               # Integration workflows (e.g. n8n automation)
├── fonts/                  # TrueType fonts for subtitle and hook rendering
├── models/                 # AI & Machine Learning weights (YOLOv8, etc.)
├── output/                 # Working directory for rendered video outputs & clips
├── remotion/               # Remotion React video composition templates
├── render-service/         # Node.js rendering worker service
├── screenshots/            # Documentation screenshots and media
├── skills/                 # AI Agent skill definitions
├── tests/                  # Pytest test suite (Unit, Integration, E2E)
├── web/                    # PeakCut reactive web studio UI, landing, and admin pages
├── app.py                  # Primary FastAPI web server & WebSocket job tracking
├── main.py                 # Primary CLI video pipeline entry point
├── colab_runner.py         # Google Colab runner & peakcut CLI console script
├── clipmind_runner.py      # Autonomous ClipMind loop runner
└── run_transform_pipeline.py # Anti-reused transformation pipeline runner
```

---

## 🛠️ Subsystems & Responsibilities

### 1. Core Processing Engine (`engine/`)
Import implementation modules from the subpackages below, for example
`from engine.editing import subtitles` or `from engine.layouts import active_speaker`.
The repository root contains project configuration and the main entry points, including
`app.py`, `main.py`, and `colab_runner.py`.

- **`engine/layouts/`**:
  - `active_speaker.py`: Dynamic face/lip speaker tracking and audio-visual correlation.
  - `camera_inset.py`: Webcam and face inset overlay compositing.
  - `layout_picker.py`: Automated AI layout selection (`split`, `screencast`, `none`).
  - `layout_ranges.py`: Time-range definitions for dynamic layout switching.
  - `panel_layout.py`: Multi-panel split-screen layouts for multi-speaker podcasts.
  - `punch_in.py`: Dynamic camera punch-ins and pacing zooms.
  - `reframe_v2.py`: 9:16 vertical reframing with subject tracking and smoothing.
  - `screencast_layout.py`: Screen share + presenter split layout.
  - `split_layout.py`: 2-shot split-screen layout with configurable seams.

- **`engine/editing/`**:
  - `branding_overlay.py`: Logo and watermark overlays with opacity and anchoring.
  - `color_grade.py`: Vertical video color presets (`vibrant_pop`, `warm_cinema`, `moody_dark`, etc.).
  - `edit_builder.py`: Complex FFmpeg filtergraph construction.
  - `editor.py`: AI-driven video filtergraph generation via Gemini.
  - `ffmpeg_utils.py`: FFmpeg command wrappers, hardware acceleration, and probes.
  - `highlight_detector.py`: Non-speech audio RMS energy envelope hype and cheer detector.
  - `hook_grounding.py`: High-retention headline timecode grounding.
  - `hooks.py`: Styled headline text overlay generation with custom fonts.
  - `recut.py`: Sub-clip trimming and boundary snapping.
  - `smartcut.py`: Jump-cutting silence and filler pauses with audio cross-fades.
  - `subtitles.py`: Word-level karaoke animations, libass burn-in, and SRT generation.
  - `thumbnail.py`: A/B viral thumbnail generation and image composition.

- **`engine/intelligence/`**:
  - `clip_selection.py`: Algorithmic clip scoring, length bounds, and deduplication.
  - `gemini_worker.py`: Google Gemini multimodal video analysis.
  - `llm_backend.py`: Unified LLM backend (Gemini, local GPU models, Ollama).
  - `topic_segmenter.py`: TextTiling transcript topic segmentation.
  - `transcribe_backends.py`: Multi-engine transcription (faster-whisper, parakeet).
  - `translate.py`: ElevenLabs AI voice translation and dubbing.
  - `viral_brain.py`: 5-axis virality scoring and platform title generation.

- **`engine/discovery/`**:
  - `channel_watcher.py`: YouTube RSS zero-quota feed poller and Twitch/Kick monitor.
  - `live_clipper.py`: Real-time live stream rolling circular buffer clipper.
  - `niche_discovery.py`: Pluggable trending and keyword discovery engine.
  - `scene_detection.py`: Shot boundary and scene transition detection.

- **`engine/services/`**:
  - `alerts_dispatcher.py`: Discord Webhook and Telegram Bot notification dispatcher.
  - `api_key_rotator.py`: Rotating API key pool with automatic rate-limit backoff.
  - `creator_workspaces.py`: Multi-tenant creator workspaces, rules, and presets.
  - `file_hosts.py`: External video and file host upload providers.
  - `profile_fetcher.py`: Channel and avatar profile metadata fetcher.
  - `publishing_queue.py`: Multi-platform scheduled publishing queue (YouTube, TikTok, IG).
  - `romanization.py`: Pure-Python phonetic transliteration for Devanagari/Urdu Hinglish.
  - `s3_uploader.py`: AWS S3 asset archiving and presigned URL delivery.
  - `system_telemetry.py`: Real-time CPU, RAM, GPU, and disk telemetry.
  - `youtube_uploader.py`: Direct YouTube Shorts OAuth uploader.

- **`engine/studio/`**:
  - `peakcut_bridge.py`: PeakCut Studio API bridge, static asset & media delivery.
  - `supercharged_routes.py`: FastAPI routes for discovery, queues, streams, and highlights.
  - `mcp_server.py`: Model Context Protocol (FastMCP) server with 13 agent tools.
  - `mcp_ui.py`: MCP inspection dashboard UI.

- **`engine/utils/`**:
  - `batch_runner.py`: Batch YouTube video automation runner.
  - `log_view.py`: Real-time log streaming and inspection.
  - `quality_probe.py`: YouTube resolution and format availability probe.
  - `saasshorts.py`: Multi-tenant SaaS billing, quota limits, and Stripe subscriptions.
  - `security_utils.py`: Path traversal validation and URL sanitization.

### 2. Autonomous Engines
- **`clipmind/`**: Autonomous trend-mining, vertical video generation, title SEO optimization, and scheduled upload loop. Run via `python clipmind_runner.py` or `python -m clipmind.cli`.
- **`anti_reused_pipeline/`**: Multi-layer video transformation (audio pitch/eq, dynamic visual framing, captions, b-roll overlays) to meet platform fair-use requirements. Run via `python run_transform_pipeline.py`.

### 3. Cloud & Multi-Tenancy (`cloud/`)
- Multi-tenant cloud features: JWT authentication, Stripe subscription billing, API key quota management, proxy ledgers, and GDPR Art. 17 account erasure.

### 4. Frontend & Rendering
- **`web/`**: PeakCut Studio reactive UI (`web/index.html`), static landing, auth, and admin pages.
- **`dashboard/`**: React + Vite single-page application with video previews, editor controls, and export tools.
- **`remotion/`**: React-based video composition engine for motion graphics, captions, and title cards.
- **`render-service/`**: Express + TypeScript service invoking Remotion bundle renderer.

### 5. Serverless & Automation (`api/`, `examples/`)
- **`api/`**: Vercel Serverless Functions (`trigger.py`, `status.py`, `cron.py`) for cloud automation.
- **`examples/n8n/`**: Ready-to-import n8n workflow templates for automated clip generation and notifications.

---

## 🚀 Key Entry Points

| Task | Command |
|---|---|
| **Run API Server** | `python -m uvicorn app:app --host 0.0.0.0 --port 8000` |
| **Run CLI Video Processing** | `python main.py -i <video_or_url> -o output/` |
| **Run PeakCut CLI** | `peakcut clip <url_or_file>` or `python colab_runner.py clip <url>` |
| **Run a batch** | `python -m engine.utils.batch_runner --file urls.txt` |
| **Discover a topic** | `python -m engine.discovery.niche_discovery --query sports` |
| **Run ClipMind Autonomous Loop** | `python clipmind_runner.py` |
| **Run Anti-Reused Pipeline** | `python run_transform_pipeline.py -i <video>` |
| **Run Full Test Suite** | `pytest` |
| **Run PeakCut UI Dev** | `python -m uvicorn app:app --reload` (open `http://localhost:8000`) |
| **Run Dashboard Frontend Dev** | `cd dashboard && npm run dev` |
| **Run Full Stack via Docker** | `docker compose up --build` |
