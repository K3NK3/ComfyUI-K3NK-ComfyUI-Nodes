"""
K3NK Save & Compress
Saves images from ComfyUI with optional compression via pngquant + oxipng.
Preserves ComfyUI metadata chunks (tEXt/iTXt/zTXt) so workflows survive drag & drop.

Binaries are auto-downloaded from your GitHub Releases on first use.
To update: upload new .exe files to a GitHub Release tagged 'binaries'.
"""

import os, struct, zlib, subprocess, tempfile, io, shutil, re, sys, json, platform
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# ── Paths ──────────────────────────────────────────────────────────────────────
_NODE_DIR = Path(__file__).resolve().parent
_BIN_DIR  = _NODE_DIR / "bin"
_BIN_DIR.mkdir(exist_ok=True)

# ── Auto-download config ───────────────────────────────────────────────────────
GITHUB_USER    = "K3NK3"
GITHUB_REPO    = "ComfyUI-K3NK-ComfyUI-Nodes"
GITHUB_TAG     = "binaries"   # your release tag

# Maps: binary stem → filename in your GitHub Release assets
_IS_WIN   = platform.system() == "Windows"
_BINARIES = {
    "pngquant": "pngquant.exe" if _IS_WIN else "pngquant",
    "oxipng":   "oxipng.exe"   if _IS_WIN else "oxipng",
}

def _download_binary(name: str, filename: str) -> str:
    """Download binary from GitHub Release if not present. Returns path or ''."""
    dest = _BIN_DIR / filename
    if dest.exists():
        return str(dest)

    url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{GITHUB_TAG}/{filename}"
    print(f"[K3NK Save] Downloading {filename} from GitHub Releases...")
    try:
        import urllib.request
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        if not _IS_WIN:
            dest.chmod(0o755)
        print(f"[K3NK Save] ✔ {filename} downloaded to {dest}")
        return str(dest)
    except Exception as e:
        print(f"[K3NK Save] ✘ Could not download {filename}: {e}")
        if tmp.exists(): tmp.unlink()
        return ""

def _get_bin(name: str) -> str:
    filename = _BINARIES.get(name, "")
    if not filename:
        return ""
    dest = _BIN_DIR / filename
    if dest.exists():
        return str(dest)
    return _download_binary(name, filename)

# Lazy-load on first use
_PNGQUANT = None
_OXIPNG   = None

def get_pngquant():
    global _PNGQUANT
    if _PNGQUANT is None:
        _PNGQUANT = _get_bin("pngquant")
    return _PNGQUANT

def get_oxipng():
    global _OXIPNG
    if _OXIPNG is None:
        _OXIPNG = _get_bin("oxipng")
    return _OXIPNG

# ── PNG chunk utils ────────────────────────────────────────────────────────────
def _read_chunks(data):
    chunks, pos = [], 8
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos+4])[0]
        t = data[pos+4:pos+8]
        d = data[pos+8:pos+8+n]
        chunks.append((t, d))
        pos += 12 + n
    return chunks

def _build_png(chunks):
    parts = [b"\x89PNG\r\n\x1a\n"]
    for t, d in chunks:
        crc = zlib.crc32(t + d) & 0xFFFFFFFF
        parts.append(struct.pack(">I", len(d)) + t + d + struct.pack(">I", crc))
    return b"".join(parts)

_META = {b"tEXt", b"iTXt", b"zTXt"}

def _extract_meta(data):
    return [(t, d) for t, d in _read_chunks(data) if t in _META]

def _inject_meta(png_bytes, meta):
    if not meta:
        return png_bytes
    chunks = _read_chunks(png_bytes)
    body   = [(t, d) for t, d in chunks if t not in _META and t != b"IEND"]
    iend   = [(t, d) for t, d in chunks if t == b"IEND"]
    return _build_png(body + meta + iend)

# ── Compressors ────────────────────────────────────────────────────────────────
def _run_tmp(fn, data):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = Path(f.name)
    try:
        tmp.write_bytes(data); fn(tmp); return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)

def _pngquant(data, qmin, qmax, speed):
    pq = get_pngquant()
    if not pq: return data
    def _r(tmp):
        out = tmp.with_name(tmp.stem + "-fs8.png")
        subprocess.run([pq, f"--quality={qmin}-{qmax}", f"--speed={speed}",
                        "--strip", "--force", "--output", str(out), str(tmp)],
                       capture_output=True, check=True)
        out.replace(tmp)
    try:
        return _run_tmp(_r, data)
    except Exception as e:
        print(f"[K3NK Save] pngquant error: {e}"); return data

def _oxipng(data, level):
    ox = get_oxipng()
    if not ox: return data
    try:
        return _run_tmp(lambda t: subprocess.run(
            [ox, f"-o{level}", "--strip", "safe", str(t)],
            capture_output=True, check=True), data)
    except Exception as e:
        print(f"[K3NK Save] oxipng error: {e}"); return data

# ── Filename helper ────────────────────────────────────────────────────────────
def _next_filename(folder: Path, base: str, ext: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in folder.glob(f"{base}_*{ext}"):
        m = re.search(r"_(\d+)_", p.name)
        if m: nums.append(int(m.group(1)))
    n = max(nums) + 1 if nums else 0
    return folder / f"{base}_{n:05d}_{ext}"

# ── Node ───────────────────────────────────────────────────────────────────────
class K3NKSaveCompress:

    COMP_MODES = ["none", "lossless (oxipng)", "lossy+lossless (pngquant+oxipng)"]
    SPEEDS     = ["1 (slowest/best)", "3 (balanced)", "6 (fast)", "10 (fastest)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":           ("IMAGE",),
                "filename_prefix":  ("STRING", {"default": "k3nk/image"}),
                "format":           (["PNG", "WEBP", "JPEG"],),
                "compression_mode": (cls.COMP_MODES,),
                "quality_min":      ("INT",  {"default": 85,  "min": 0,   "max": 100}),
                "quality_max":      ("INT",  {"default": 100, "min": 0,   "max": 100}),
                "pngquant_speed":   (cls.SPEEDS,),
                "oxipng_level":     ("INT",  {"default": 2,   "min": 1,   "max": 6}),
                "lossy_quality":    ("INT",  {"default": 90,  "min": 1,   "max": 100}),
            },
            "hidden": {
                "prompt":        "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION     = "save"
    OUTPUT_NODE  = True
    CATEGORY     = "K3NK/IO"

    def save(self, images, filename_prefix, format, compression_mode,
             quality_min, quality_max, pngquant_speed, oxipng_level, lossy_quality,
             prompt=None, extra_pnginfo=None):

        import folder_paths
        output_dir  = Path(folder_paths.get_output_directory())
        prefix_path = Path(filename_prefix)
        save_folder = output_dir / prefix_path.parent
        base_name   = prefix_path.name or "image"
        speed_val   = int(pngquant_speed.split(" ")[0])
        ext_map     = {"PNG": ".png", "WEBP": ".webp", "JPEG": ".jpg"}
        ext         = ext_map[format]
        saved       = []

        for img_tensor in images:
            arr = (img_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil = Image.fromarray(arr)

            # Build PngInfo with ComfyUI metadata
            pnginfo = PngInfo()
            if extra_pnginfo:
                for k, v in extra_pnginfo.items():
                    pnginfo.add_text(k, json.dumps(v) if not isinstance(v, str) else v)
            if prompt:
                pnginfo.add_text("prompt", json.dumps(prompt))

            out_path = _next_filename(save_folder, base_name, ext)

            if format == "PNG":
                buf = io.BytesIO()
                pil.save(buf, format="PNG", pnginfo=pnginfo, optimize=True, compress_level=9)
                raw  = buf.getvalue()
                meta = _extract_meta(raw)

                if compression_mode == "lossless (oxipng)":
                    o = _oxipng(raw, oxipng_level)
                    if len(o) < len(raw): raw = o

                elif compression_mode == "lossy+lossless (pngquant+oxipng)":
                    q = _pngquant(raw, quality_min, quality_max, speed_val)
                    if len(q) < len(raw): raw = q
                    o = _oxipng(raw, oxipng_level)
                    if len(o) < len(raw): raw = o

                raw = _inject_meta(raw, meta)
                out_path.write_bytes(raw)

            elif format == "WEBP":
                pil.save(str(out_path), format="WEBP", quality=lossy_quality, method=6)

            elif format == "JPEG":
                if pil.mode == "RGBA": pil = pil.convert("RGB")
                pil.save(str(out_path), format="JPEG", quality=lossy_quality, optimize=True)

            saved.append(str(out_path))
            size_kb = out_path.stat().st_size // 1024
            print(f"[K3NK Save] ✔ {out_path.name}  ({size_kb} KB)")

        return {"ui": {"images": [
            {"filename": Path(p).name,
             "subfolder": str(prefix_path.parent),
             "type": "output"} for p in saved
        ]}}


NODE_CLASS_MAPPINGS        = {"K3NKSaveCompress": K3NKSaveCompress}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKSaveCompress": "K3NK Save & Compress 🗜️"}
