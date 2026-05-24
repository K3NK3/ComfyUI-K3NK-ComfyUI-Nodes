# K3NK ComfyUI Nodes

A collection of ComfyUI custom nodes for advanced video workflows, image processing, and latent management.

---

## Nodes

### 🖼️ K3NK Image Loader with Blending
**Category:** `K3NK/loaders`

Loads image sequences from a directory with **smootherstep blending** at sequence boundaries — designed for multi-batch video workflows (WanVideo, AnimateDiff, etc.) where re-generated segments need seamless transitions.

| Input | Type | Description |
|---|---|---|
| `directory_path` | STRING | Path to folder with image sequence |
| `sequence_frames` | INT | Frames per sequence (default: 81) |
| `overlap_frames` | INT | Frames to blend at boundaries (default: 5) |
| `file_pattern` | STRING | Glob pattern, e.g. `*.png` |

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Blended tensor `[N, H, W, 3]` |

**Notes:** Output has fewer frames than input — blending replaces overlap frames, not adds them. For RIFE 2× interpolation, double both `sequence_frames` and `overlap_frames`.

---

### 🎬 K3NK Image Grab (Last N Frames)
**Category:** `K3NK/loaders`

Grabs the **last N frames** from a directory as a single batch for video continuation. Supports optional frame stride to skip frames between selections, improving temporal consistency with interpolated frames.

| Input | Type | Description |
|---|---|---|
| `directory` | STRING | Path to image directory |
| `frame_count` | INT | Number of frames to grab from the end |
| `stride` | INT | Step between selected frames (1 = every frame) |
| `file_pattern` | STRING | Glob pattern (default: `*.png`) |

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Batch tensor of last N frames |

**Use case:** Feed into WanVideoWrapper ClipVision Encode or other video continuation nodes.

---

### 🎬 Load Video Batch From Dir (K3NK)
**Category:** `K3NK/loaders`

Loads a single video from a directory by index, with flexible sorting and frame extraction via PyAV. Returns frames as an IMAGE batch compatible with any ComfyUI video node.

| Input | Type | Description |
|---|---|---|
| `directory` | STRING | Path to folder with video files |
| `sort_method` | ENUM | None / Alphabetical / Numerical / Datetime (ASC/DESC) |
| `start_index` | INT | Which video to load (0-based) |
| `frame_load_cap` | INT | Max frames to extract (0 = all) |
| `load_always` | BOOLEAN | Force re-execute every run |

| Output | Type | Description |
|---|---|---|
| `images` | IMAGE | Frames as `[N, H, W, 3]` tensor |
| `masks` | MASK | Empty masks `[N, H, W]` |
| `frame_count` | INT | Number of frames loaded |
| `video_info` | VHS_VIDEOINFO | FPS, resolution, duration metadata |

**Supported formats:** mp4, avi, mov, mkv, webm, flv, wmv, m4v, 3gp

---

### 📄 Load Text From Dir (K3NK)
**Category:** `K3NK/loaders`

Loads a `.txt` file from a directory by index, with sorting and encoding options. Useful for iterating through prompt files in batch workflows.

| Input | Type | Description |
|---|---|---|
| `directory` | STRING | Path to folder with `.txt` files |
| `sort_method` | ENUM | None / Alphabetical / Numerical / Datetime (ASC/DESC) |
| `start_index` | INT | Which file to load (0-based) |
| `encoding` | ENUM | utf-8 / utf-8-sig / latin-1 / cp1252 |
| `load_always` | BOOLEAN | Force re-execute every run |

| Output | Type | Description |
|---|---|---|
| `text` | STRING | Full file contents |
| `index` | INT | Current index |
| `total_files` | INT | Total `.txt` files found |
| `filename` | STRING | Loaded filename |

---

### 💾 Save Latent (Pass-Through)
**Category:** `K3NK/IO`

Saves latents to disk mid-workflow **without interrupting the pipeline**. Ideal for checkpointing long generation processes.

| Input | Type | Description |
|---|---|---|
| `samples` | LATENT | Latent to save |
| `filename_prefix` | STRING | Prefix, supports subfolders: `subdir/name` |

| Output | Type | Description |
|---|---|---|
| `samples` | LATENT | Same latent, unmodified |

Files saved as `{prefix}_{00000}_.latent` in ComfyUI's output directory.

---

### 🗜️ K3NK Save & Compress
**Category:** `K3NK/IO`

Saves images with optional compression (like TinyPNG) while **preserving ComfyUI workflow metadata** — images can be drag & dropped back into ComfyUI with their workflow intact.

Binaries are auto-downloaded from GitHub Releases on first use.

| Input | Type | Description |
|---|---|---|
| `images` | IMAGE | Images to save |
| `filename_prefix` | STRING | Save path, supports subfolders: `myfolder/image` |
| `format` | ENUM | PNG / WEBP / JPEG |
| `compression_mode` | ENUM | none / lossless (oxipng) / lossy+lossless (pngquant+oxipng) |
| `quality_min` | INT | pngquant min quality 0-100 (default: 85) |
| `quality_max` | INT | pngquant max quality 0-100 (default: 100) |
| `pngquant_speed` | ENUM | 1 (slowest/best) → 10 (fastest) |
| `oxipng_level` | INT | Optimization level 1-6 (default: 2) |
| `lossy_quality` | INT | Quality for WEBP/JPEG 1-100 (default: 90) |

**Compression results (typical Flux images):**
- `none` → baseline
- `lossless (oxipng)` → ~5-10% smaller, pixel-perfect
- `lossy+lossless (pngquant+oxipng)` → ~50-70% smaller, visually identical

**Third-party binaries** (auto-downloaded, no authorship claimed):
- pngquant — Kornel Lesiński, GPL v3 — https://pngquant.org
- oxipng — Joshua Holmer, MIT — https://github.com/oxipng/oxipng

---

### 🖼️ K3NK Image Overlay
**Category:** `K3NK/image`

Memory-efficient image overlay node — functionally identical to TSC Image Overlay from efficiency-nodes-comfyui but without PIL conversions, significantly reducing peak VRAM/RAM usage on large batches.

| Input | Type | Description |
|---|---|---|
| `base_image` | IMAGE | Background image batch |
| `overlay_image` | IMAGE | Image to composite on top |
| `overlay_resize` | ENUM | None / Fit / Resize by factor / Resize to W×H |
| `resize_method` | ENUM | nearest-exact / bilinear / area |
| `rescale_factor` | FLOAT | Scale multiplier (default: 1.0) |
| `width` / `height` | INT | Target size when resizing |
| `x_offset` / `y_offset` | INT | Position of overlay |
| `rotation` | INT | Degrees (-180 to 180) |
| `opacity` | FLOAT | 0 = fully opaque, 100 = invisible |
| `optional_mask` | MASK | White = transparent, black = opaque |

| Output | Type | Description |
|---|---|---|
| `image` | IMAGE | Composited batch |

---

### 🎬 K3NK Video Concat Simple
**Category:** `K3NK/video-tools`

Concatenates multiple `.mp4` videos from a directory using FFmpeg. Supports re-encode or copy stream mode.

| Input | Type | Description |
|---|---|---|
| `directory_path` | STRING | Folder with video files |
| `output_filename` | STRING | Output filename (no extension) |
| `output_format` | ENUM | mp4 / avi / webm / mov / mkv / gif |
| `output_absolute_path` | STRING | Optional absolute output path |
| `reencode` | BOOLEAN | Re-encode (slow, compatible) or copy stream (fast) |

| Output | Type | Description |
|---|---|---|
| `output_path` | STRING | Full path to concatenated video |

**Note:** WebM does not support copy stream (VP9+Opus required) — falls back to MP4 automatically.

---

### 📐 K3NK Find Nearest Bucket
**Category:** `K3NK/utils`

Finds the nearest model-compatible resolution bucket for a given image size. Useful for feeding images into video models that require specific aspect ratios.

---

## Installation

```
ComfyUI/custom_nodes/
└── ComfyUI-K3NK-ComfyUI-Nodes/
```

Clone or download into `custom_nodes/` and restart ComfyUI.

**Requirements:** `pip install pillow` (most others come with ComfyUI)

For video nodes: `pip install av`

---

## Version History

- **v1.8**: Added K3NK Save & Compress with pngquant+oxipng pipeline and auto-download binaries
- **v1.7**: Added Load Video Batch From Dir and Load Text From Dir
- **v1.6**: Added K3NK Image Overlay, memory-efficient drop-in for TSC Image Overlay
- **v1.5**: Added K3NK Find Nearest Bucket
- **v1.4**: Added Save Latent (Pass-Through)
- **v1.3**: Switched to smootherstep blending in Image Loader
- **v1.0**: Initial release

---

## License

MIT — free to use and modify.

Designed for WanVideo, AnimateDiff, RIFE, and multi-batch video workflows.
