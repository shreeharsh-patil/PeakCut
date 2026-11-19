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
├── examples/               # Integration workflows (e.g. n8n automation)
├── fonts/                  # TrueType fonts for subtitle and hook rendering
├── models/                 # AI & Machine Learning weights (YOLOv8, etc.)
├── output/                 # Working directory for rendered video outputs & clips
├── PeakCut/                # PeakCut Studio backend & reactive UI
├── remotion/               # Remotion React video composition templates
├── render-service/         # Node.js rendering worker service
├── screenshots/            # Documentation screenshots and media
├── skills/                 # AI Agent skill definitions
├── tests/                  # Pytest test suite (Unit, Integration, E2E)
├── web/                    # Static landing, auth, and admin pages
└── Core Engine (Root)      # Core Python video processing & FastAPI server
```

---

## 🛠️ Subsystems & Responsibilities

### 1. Core Processing Engine (`./`)
The root directory houses the primary Python video pipeline:
- **`app.py`**: FastAPI application exposing REST endpoints, job tracking, and WebSocket status.
- **`main.py`**: CLI video processing engine: ingestion, transcription, scene detection, vertical reframing, and FFmpeg rendering.
- **`peakcut_bridge.py`**: Compatibility bridge connecting OpenShorts backend with PeakCut studio.
- **`publishing_queue.py`**: Persistent queue managing scheduled publishing across platforms (YouTube, etc.).
- **`subtitles.py`**: Word-level subtitle generation, karaoke animations, and libass burn-in.
- **`hooks.py`**: High-retention headline and hook generation with shadow and font styling.
- **`active_speaker.py` / `reframe_v2.py`**: Dynamic speaker tracking and vertical camera panning.
- **`smartcut.py` / `recut.py`**: Silence cutting, filler word elimination, and pacing optimization.
- **`thumbnail.py`**: High-CTR A/B thumbnail generator.
- **`translate.py`**: AI voice translation and dubbing.

### 2. Autonomous Engines
- **`clipmind/`**: Autonomous trend-mining, vertical video generation, title SEO optimization, and scheduled upload loop. Run via `python clipmind_runner.py` or `python -m clipmind.cli`.
- **`anti_reused_pipeline/`**: Multi-layer video transformation (audio pitch/eq, dynamic visual framing, captions, b-roll overlays) to meet platform fair-use requirements. Run via `python run_transform_pipeline.py`.

### 3. Cloud & Multi-Tenancy (`cloud/`)
- Multi-tenant cloud features: JWT authentication, Stripe subscription billing, API key quota management, proxy ledgers, and GDPR Art. 17 account erasure.

### 4. Frontend & Rendering
- **`dashboard/`**: React + Vite single-page application with video previews, editor controls, and export tools.
- **`remotion/`**: React-based video composition engine for motion graphics, captions, and title cards.
- **`render-service/`**: Express + TypeScript service invoking Remotion bundle renderer.
- **`web/`**: Fast static landing and authentication pages.

### 5. Serverless & Automation (`api/`, `examples/`)
- **`api/`**: Vercel Serverless Functions (`trigger.py`, `status.py`, `cron.py`) for cloud automation.
- **`examples/n8n/`**: Ready-to-import n8n workflow templates for automated clip generation and notifications.

### 6. Machine Learning Weights (`models/`)
- **`models/`**: Dedicated folder for local weights (such as `models/yolov8n.pt`) separated from the root.

---

## 🚀 Key Entry Points

| Task | Command |
|---|---|
| **Run API Server** | `python -m uvicorn app:app --host 0.0.0.0 --port 8000` |
| **Run CLI Video Processing** | `python main.py -i <video_or_url> -o output/` |
| **Run ClipMind Autonomous Loop** | `python clipmind_runner.py` |
| **Run Anti-Reused Pipeline** | `python run_transform_pipeline.py -i <video>` |
| **Run Full Test Suite** | `pytest` |
| **Run PeakCut Suite** | `python PeakCut/tests/run.py` |
| **Run Frontend Dev Server** | `cd dashboard && npm run dev` |
| **Run Full Stack via Docker** | `docker compose up --build` |
