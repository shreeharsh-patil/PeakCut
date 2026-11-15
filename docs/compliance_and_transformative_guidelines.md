# YouTube Platform Compliance, Fingerprinting Mechanics & Transformative Content Guidelines

> **Document Version:** 1.0.0  
> **Target Audience:** Engineering (`@builder`), Quality & Review (`@verifier`), Orchestration (`@coordinator`)  
> **Scope:** OpenShorts / PeakCut Production Pipeline (`D:\yt`)

---

## 1. Executive Summary & Purpose

Automated video repurposing (e.g., generating YouTube Shorts, TikToks, and Reels from longer third-party or source media) faces two distinct platform enforcement barriers:
1. **Tier 1: Algorithmic Ingestion Enforcement** (YouTube Content ID & Copyright Match Tool).
2. **Tier 2: Monetization & Policy Review** (YouTube Partner Program Reused & Repetitious Content Classifiers + Human Quality Reviewers).

A widespread industry pitfall is **"Superficial Cloaking"** (commonly known in Chinese automation circles as *去重* / *qùchóng*). These techniques rely on micro-distortions (micro-zooming, 1.02x speed shifts, mirroring, imperceptible noise, border masks). While they may slightly alter naive perceptual hashes, they **fail completely against modern multi-modal neural fingerprinting** and **trigger automatic rejection or demonetization during YPP channel review**.

To achieve sustainable, monetizable automation, the pipeline must implement **True Automated Transformative Value**. This guide defines the technical mechanics of Content ID, details the boundary between cloaking and genuine transformation, and provides concrete engineering criteria for `@builder`.

---

## 2. YouTube Content ID & Multi-Modal Fingerprinting Mechanics

### 2.1 Audio Acoustic Fingerprinting
Content ID audio matching operates on a dual-engine paradigm: classical acoustic constellation hashing and modern neural audio representations.

#### A. Classical Constellation Hashing (Peak Extraction)
1. **Spectrogram Transformation:** The input audio stream is converted into the frequency-time domain via Short-Time Fourier Transform (STFT):
   $$X(t, f) = \int_{-\infty}^{\infty} x(\tau) w(\tau - t) e^{-j 2\pi f \tau} d\tau$$
2. **Landmark / Peak Detection:** Local energy maxima (spectral peaks where amplitude exceeds adjacent time-frequency bins) are extracted across frequency bands.
3. **Combinatorial Hashing:** Pairs of peaks separated by a time delta $\Delta t$ form discrete hash tokens:
   $$\text{Hash} = [f_1 \parallel f_2 \parallel \Delta t]$$
4. **Time-Series Co-Linear Alignment:** Query hashes are matched against Google's inverted index. Content ID does not require an exact match of the entire file; a match is flagged if as few as **3.0 to 5.0 seconds** of continuous hash tokens exhibit identical chronological slope (temporal alignment via Dynamic Time Warping / Hough transform).

#### B. Neural Audio Embeddings (AST / VGGish)
* Google deploys Audio Spectrogram Transformers (AST) and deep convolutional acoustic models.
* These models compress 1-second audio windows into dense 128-d or 512-d semantic embedding vectors.
* **Resilience:** Resistant to simple pitch-shifting ($\pm 10\%$), tempo stretching, low-pass/high-pass EQ carving, and linear re-sampling. If the underlying melodic/harmonic progression or voice timbre matches the reference embedding vector space, candidate matches are triggered.

#### C. What Defeats vs. What Fails in Audio
* ❌ **Fails (Superficial):** $\pm 3\%$ pitch shift, minor tempo adjustments (1.02x), adding low-volume pink/white noise, stereo channel inversion. The constellation peaks remain prominently detectable.
* ✅ **Succeeds (Transformative):** 
  - **Dominant Original Voiceover:** Replacing or aggressively ducking (-18 dB to -24 dB) the source audio beneath an original, continuous commentary track.
  - **Spectral Carving & Sidechain Compression:** Compressing source audio dynamic range and notching source frequencies during speech.
  - **Complete Audio Track Replacement:** Muting source audio entirely and driving the Short with original synthetic voiceover (TTS) plus licensed background music.

---

### 2.2 Video Perceptual Hashing & Spatial-Temporal Feature Matching

Content ID's visual matching engine scans video streams through a multi-resolution hierarchy:

#### A. Spatio-Temporal Segmentation & Perceptual Hashing (pHash / dHash)
1. **Shot Boundary Detection:** The video is partitioned into discrete scenes using temporal color-histogram and edge-change differences.
2. **Perceptual Downsampling:** Frames are converted to grayscale and downscaled to normalized matrices (e.g., $32 \times 32$ or $64 \times 64$).
3. **Discrete Cosine Transform (DCT):** Low-frequency components (representing broad shapes, luminances, and compositions) are retained; high-frequency noise is discarded.
4. **Hamming Distance Verification:** Binary perceptual hashes are indexed using Locality-Sensitive Hashing (LSH). Videos with normalized Hamming distance $D_H \le 0.12$ trigger candidates.

#### B. Deep Visual Embeddings & Invariant Descriptors
* **Deep Models (ViViT / VideoBERT / CLIP-like):** Google models frame sequences into dense spatial-temporal tokens.
* **Geometric Invariance:** The system automatically checks both normal and horizontally mirrored orientations.
* **Transform Tolerance:** Content ID readily handles:
  - Linear scaling and aspect-ratio padding.
  - Modest center cropping (up to 15-20%).
  - Global color adjustments (brightness, contrast, saturation, LUTs).
  - Re-encoding compression artifacts (H.264/HEVC/AV1 at varying bitrates).

#### C. What Defeats vs. What Fails in Video Matching
* ❌ **Fails (Superficial):** 
  - 1.02x–1.05x micro-zoom (DCT low-frequency matrix remains $>85\%$ correlated).
  - Subtle vignette or color tinting (grayscale DCT strips color; vignette affects only peripheral bins).
  - Horizontal flip alone (Content ID dual-indexes mirrored matrices).
  - Micro-frame rate change (29.97 to 30.0 fps or minor frame decimation).
* ✅ **Succeeds (Transformative Restructuring):**
  - **Layout Re-composition:** Multi-window split screen (e.g., 50/50 or 60/40 vertical split), Picture-in-Picture (reaction face/avatar overlay occupying $\ge 25\%$ canvas area).
  - **Temporal Slicing & Interleaving:** Restricting raw continuous source segments to $\le 5$ seconds, interspersed with freeze-frame analysis, B-roll, or graphics.
  - **Substantial Canvas Occlusion & Annotation:** Dynamic kinetic typography, highlight overlays, contextual callouts, and blurred backdrop frames.

---

### 2.3 Copyright Match Tool vs. Content ID
* **Content ID:** Enterprise tool for major rights holders (labels, movie studios, major networks). Automatically blocks, tracks, or claims ad revenue on matching uploads.
* **Copyright Match Tool:** Available to all YPP creators and channels with original uploads. Continuously checks the entire YouTube catalog for identical or near-identical re-uploads of public videos. If a creator uploads an unedited snippet from another creator, the original creator receives a notification with a percentage match (e.g., "95% match") and can issue a takedown with one click.

---

## 3. YouTube Partner Program (YPP) Reused & Repetitive Content Detection

Even if a video completely evades Content ID matches (e.g., source media without an active enterprise fingerprint), it must pass **Tier 2 Monetization Compliance**.

### 3.1 The Reused Content Policy
> **YouTube Official Policy:**  
> *"Reused content refers to channels that repurpose someone else's content without adding significant original commentary or educational value. If you upload content from multiple sources or repurpose existing content, you must add your own distinct creative voice or editorial narrative."*

#### Automated Detection Mechanics for Reused Content
1. **Automatic Speech Recognition (ASR) & Semantic Hashing:**  
   Google runs multi-lingual Whisper/Google Speech models across all uploads to generate full text transcripts.
   - Transcripts are embedded into vector space and compared against YouTube's speech database.
   - If an uploaded video's transcript has $>70\%$ semantic overlap with an existing viral video, lecture, or podcast, it is flagged as reused content.
2. **Template Repetition Classifiers:**  
   Detects programmatic, mass-produced formats:
   - Identical background music loops across dozens of videos.
   - Monotonous, flat robotic TTS voices without prosodic variation.
   - Static template overlays where only centered subtitles change.
3. **Human Quality Review:**  
   During initial YPP application and periodic channel audits, human evaluators review:
   - Most viewed videos.
   - Most recent uploads.
   - Significant sources of watch time.
   - Channel branding, titles, and descriptions.
   - Evaluator Question: *"Did this creator personally synthesize, explain, react to, or transform this media, or did they simply package someone else's work?"*

---

## 4. The Strict Boundary: Cloaking vs. True Transformation

| Transformation Dimension | Superficial Cloaking (Reused / Demonetized) | True Automated Transformative Value (Monetizable / Fair Use) |
| :--- | :--- | :--- |
| **Audio Architecture** | Original audio kept at 100%; pitch shifted $\pm 2$ semitones; slight reverb or background noise added. | Original audio ducked to $-24\text{ dB}$ or muted; original continuous voiceover (human or neural TTS) drives narrative ($>70\%$ time). |
| **Visual Composition** | Full-screen 9:16 crop with micro-zoom (1.03x), subtle edge blur, or minor color grading filter. | Composite layout: Split-screen (video + speaker/b-roll), PIP commentary avatar, blurred background frame, graphic callout panels. |
| **Temporal Editing** | Full 30–60 second continuous scene copied directly from source with no internal cuts. | Segmented curation: Max single clip duration $\le 5$s; clips re-sequenced, interleaved with pause frames, reaction beats, or B-roll. |
| **Narrative Input** | No voiceover; or robotic AI voice reading verbatim transcript of the source video. | Original script providing criticism, analysis, explanation, humor, or educational perspective. Unique transcript. |
| **Text & Graphics** | Basic auto-generated subtitles centered over raw video; generic static borders. | Kinetic word-highlight captions, infographic annotations, arrows/circles highlighting specific visual evidence. |
| **Legal / Fair Use Alignment** | No new expression; purely derivative; free-riding on original creator's audience. | Satisfies Campbell v. Acuff-Rose: Creates a new meaning, message, or critical understanding of the source material. |

---

## 5. Engineering Specifications for Pipeline Implementation (`@builder`)

The pipeline in `D:\yt` must strictly enforce the following architectural layers:

### 5.1 Audio Transformation Engine (The Ducking & Primacy Rule)
1. **Voiceover Dominance:**
   - Every repurposed clip MUST integrate an original voiceover track (`voiceover.mp3` or TTS generated).
   - Voiceover must occupy at least **70% of total video duration**.
2. **Automated Sidechain Ducking Filter:**
   When voiceover is present, source audio must be attenuated by $-20\text{ dB}$ to $-26\text{ dB}$.
   - **FFmpeg Filtergraph:**
     ```bash
     [1:a]asplit=2[sc][vo];
     [0:a][sc]sidechaincompress=threshold=0.08:ratio=10:attack=15:release=300[ducked];
     [ducked][vo]amix=inputs=2:weights=0.2 1.0[mixed_audio];
     [mixed_audio]loudnorm=I=-14:TP=-2.0:LRA=11[aout]
     ```
   - Target overall loudness: **$-14\text{ LUFS}$**, True Peak **$-2.0\text{ dBTP}$** (via existing `ffmpeg_utils.LOUDNORM_FILTER`).
3. **Source Audio Mute / Replacement Option:**
   - For high-risk source material (music videos, movies, TV broadcasts), provide a `mute_source=True` flag that replaces source audio entirely with synthetic voiceover + royalty-free background bed.

---

### 5.2 Visual Layout Engine (Multi-Layer Restructuring)
The pipeline must support four compliant layout modes for vertical 9:16 ($1080\times1920$) output:

1. **Split-Screen Layout (Commentary / Dual-Stream):**
   - Top panel ($1080\times960$): Source clip, cropped/scaled to fit 1:1 or 16:9 box.
   - Bottom panel ($1080\times960$): Secondary video (commentator facecam, animated avatar, or context footage).
   - Center divider: Branded accent line ($4\text{px}$) with channel badge.
2. **Picture-in-Picture (PIP Reaction Inset):**
   - Canvas: Scaled source video with blurred background padding (`boxblur=20:5`).
   - Inset: Speaker / Reaction circle or rounded rectangle occupying **$25\%\text{--}35\%$** of bottom/top canvas area with subtle border and drop shadow.
3. **Letterboxed Educational Canvas:**
   - Source clip centered at native aspect ratio ($1080\times608$ for 16:9).
   - Upper canvas ($1080\times600$): Dynamic topic headline, question, or context card.
   - Lower canvas ($1080\times712$): Kinetic subtitle transcription and analytical commentary notes.
4. **Active Graphic Annotations:**
   - Timed highlight overlays (bounding boxes, arrows, zoom-ins to detail) during key analytical remarks.

---

### 5.3 Temporal Slicing & Curation Engine
1. **Maximum Continuous Chunk Rule ($T_{\text{chunk}} \le 5.0\text{s}$):**
   - No single unedited segment of the source video may play uninterrupted for more than **5 seconds**.
   - After each chunk, the engine must execute one of:
     * Hard cut to next curated moment.
     * Freeze frame with audio zoom / spotlight effect for commentary.
     * Cutaway to commentary speaker or illustrative B-roll.
2. **Non-Linear Re-Sequencing:**
   - Where applicable, order clips by narrative impact rather than straight chronological playback.

---

### 5.4 Metadata Sanitation & Transcode Cleanliness
To prevent container-level tracking and metadata leakage:
1. Strip all source container and stream metadata tags:
   `-map_metadata -1 -map_chapters -1 -map_metadata:s:v -1 -map_metadata:s:a -1` (see `ffmpeg_utils.METADATA_SCRUB`).
2. Standardize color space:
   `-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv`.
3. Normalize video stream parameters:
   `-r 30 -pix_fmt yuv420p -movflags +faststart`.
4. Ensure clean stream timing (Constant Frame Rate, reset presentation timestamps `setpts=PTS-STARTPTS`, `asetpts=PTS-STARTPTS`).

---

## 6. Programmatic Compliance Scoring Metric ($S_{\text{compliance}}$)

`@builder` should incorporate an automated verification check that calculates an objective **Transformative Compliance Score ($S$)** between $0.0$ and $1.0$:

$$S = 0.35 \cdot R_{\text{VO}} + 0.25 \cdot R_{\text{layout}} + 0.25 \cdot R_{\text{temporal}} + 0.15 \cdot R_{\text{audio\_duck}}$$

Where:
* $R_{\text{VO}} = \min\left(1.0, \frac{\text{Voiceover Duration}}{\text{Total Video Duration}}\right)$  
  *(Target: $\ge 0.70$)*
* $R_{\text{layout}} = 1.0$ if Split-Screen or PIP Inset $\ge 25\%$ canvas; $0.5$ if Canvas Letterbox with Headers/Subtitles; $0.0$ if raw source fills full 9:16 frame.
* $R_{\text{temporal}} = \begin{cases} 1.0 & \text{if } \max(T_{\text{chunk}}) \le 5.0\text{s} \\ 0.6 & \text{if } \max(T_{\text{chunk}}) \le 8.0\text{s} \\ 0.0 & \text{if } \max(T_{\text{chunk}}) > 8.0\text{s} \end{cases}$
* $R_{\text{audio\_duck}} = 1.0$ if source audio is ducked $\ge 18\text{ dB}$ or muted; $0.0$ if source audio runs at parity.

### Threshold Gate:
* **$S \ge 0.80$**: ✅ **PASS (Compliant & Monetizable)**. Ready for publication.
* **$0.60 \le S < 0.80$**: ⚠️ **WARNING (Borderline)**. Requires manual verification; vulnerable to Content ID if high-profile rights holder.
* **$S < 0.60$**: ❌ **FAIL (High Risk / Reused Content)**. Pipeline should halt or flag before export.

---

## 7. Direct Instructions for Swarm Roles

* **To `@builder`:**
  - Implement the audio sidechain ducking filter and loudness normalization in the audio assembly module.
  - Provide modular layout presets: Split-Screen, PIP Inset, and Canvas Letterbox.
  - Implement the chunk duration slicer ($T_{\text{chunk}} \le 5\text{s}$) in clip ingestion/editing.
  - Embed the `calculate_compliance_score()` validator function into the output export pipeline.
* **To `@verifier`:**
  - Audit all test renders against the $S \ge 0.80$ compliance metric.
  - Inspect exported files with `ffprobe` to ensure metadata is stripped and audio levels match $-14\text{ LUFS} \pm 1$.
  - Confirm that no test clip features unedited source footage exceeding $5.0$ seconds.
