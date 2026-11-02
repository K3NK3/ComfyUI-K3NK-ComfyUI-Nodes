"""
K3NK Save & Compress - ULTRA ROBUSTO V3 + Civitai metadata
"""

import os, struct, zlib, subprocess, tempfile, io, shutil, re, sys, json, platform
from pathlib import Path

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    import piexif
except ImportError:
    piexif = None

_NODE_DIR = Path(__file__).resolve().parent
_BIN_DIR  = _NODE_DIR / "bin"
_BIN_DIR.mkdir(exist_ok=True)

GITHUB_USER, GITHUB_REPO, GITHUB_TAG = "K3NK3", "ComfyUI-K3NK-ComfyUI-Nodes", "binaries"
_IS_WIN   = platform.system() == "Windows"
_BINARIES = {"pngquant": "pngquant.exe" if _IS_WIN else "pngquant",
             "oxipng":   "oxipng.exe"   if _IS_WIN else "oxipng"}

def _download_binary(name, filename):
    dest = _BIN_DIR / filename
    if dest.exists(): return str(dest)
    url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{GITHUB_TAG}/{filename}"
    try:
        import urllib.request
        tmp = dest.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
        if not _IS_WIN: dest.chmod(0o755)
        return str(dest)
    except Exception as e:
        print(f"[K3NK Save] no se pudo descargar {filename}: {e}")
        return ""

def _get_bin(name):
    filename = _BINARIES.get(name, "")
    if not filename: return ""
    dest = _BIN_DIR / filename
    return str(dest) if dest.exists() else _download_binary(name, filename)

_PNGQUANT = _OXIPNG = None
def get_pngquant():
    global _PNGQUANT
    if _PNGQUANT is None: _PNGQUANT = _get_bin("pngquant")
    return _PNGQUANT
def get_oxipng():
    global _OXIPNG
    if _OXIPNG is None: _OXIPNG = _get_bin("oxipng")
    return _OXIPNG

def _read_chunks(data):
    chunks, pos = [], 8
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos+4])[0]
        t = data[pos+4:pos+8]
        d = data[pos+8:pos+8+n]
        chunks.append((t, d)); pos += 12 + n
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
    if not meta: return png_bytes
    chunks = _read_chunks(png_bytes)
    body = [(t, d) for t, d in chunks if t not in _META and t != b"IEND"]
    iend = [(t, d) for t, d in chunks if t == b"IEND"]
    return _build_png(body + meta + iend)

import threading, queue as _queue

_compress_queue = _queue.Queue()

def _compress_worker():
    while True:
        args = _compress_queue.get()
        try:
            _compress_async(*args)
        finally:
            _compress_queue.task_done()

threading.Thread(target=_compress_worker, daemon=True).start()

def _compress_async(path, compression_mode, quality_min, quality_max, speed_val, oxipng_level):
    try:
        size_before = path.stat().st_size
        pending = _compress_queue.qsize()
        print(f"[K3NK Save] ⏳ comprimiendo {path.name} ({size_before//1024} KB) | pendientes en cola: {pending}")
        raw = path.read_bytes()
        meta = _extract_meta(raw)
        if compression_mode == "optimize (PIL)":
            img = Image.open(io.BytesIO(raw))
            b2 = io.BytesIO()
            img.save(b2, format="PNG", optimize=True, compress_level=9)
            o = b2.getvalue()
            if len(o) < len(raw): raw = o
        elif compression_mode == "lossless (oxipng)":
            o = _oxipng(raw, oxipng_level)
            if len(o) < len(raw): raw = o
        elif compression_mode == "lossy+lossless (pngquant+oxipng)":
            q = _pngquant(raw, quality_min, quality_max, speed_val)
            if len(q) < len(raw): raw = q
            o = _oxipng(raw, oxipng_level)
            if len(o) < len(raw): raw = o
        raw = _inject_meta(raw, meta)
        path.write_bytes(raw)
        size_after = len(raw)
        pct = 100 - (size_after * 100 // size_before)
        print(f"[K3NK Save] ✔ (bg) {path.name}: {size_before//1024} KB → {size_after//1024} KB (-{pct}%)")
    except Exception as e:
        print(f"[K3NK Save] ✘ error compresión bg: {e}")

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
    qmax = min(qmax, 99)
    if qmin >= qmax: qmin = max(0, qmax - 10)
    def _r(tmp):
        out = tmp.with_name(tmp.stem + "-fs8.png")
        r = subprocess.run([pq, f"--quality={qmin}-{qmax}", f"--speed={speed}",
                        "--strip", "--force", "--output", str(out), str(tmp)],
                       capture_output=True)
        if r.returncode == 0: out.replace(tmp)
    try: return _run_tmp(_r, data)
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

# ── Civitai metadata helpers ─────────────────────────────────────────────────
def _find_first(prompt_dict, class_types):
    for node in prompt_dict.values():
        if node.get("class_type") in class_types:
            return node
    return None

def _build_a1111_string(prompt_dict, positive_text="", negative_text=""):
    pos, neg = positive_text, negative_text
    steps, cfg, sampler, seed, model = 20, 8, "euler", 0, ""
    ks = _find_first(prompt_dict, {"KSampler", "KSamplerAdvanced"})
    if ks:
        i = ks.get("inputs", {})
        steps = i.get("steps", steps); cfg = i.get("cfg", cfg)
        sampler = i.get("sampler_name", sampler)
        seed = i.get("seed", i.get("noise_seed", seed))
    ck = _find_first(prompt_dict, {"CheckpointLoaderSimple", "CheckpointLoader"})
    if ck:
        model = ck.get("inputs", {}).get("ckpt_name", model)
    return (f"{pos}\nNegative prompt: {neg}\n"
            f"Steps: {steps}, Sampler: {sampler}, CFG scale: {cfg}, "
            f"Seed: {seed}, Model: {model}")

def _exif_bytes(params_str):
    if not piexif: return None
    comment = params_str.encode("utf-16le")
    exif_dict = {"Exif": {piexif.ExifIFD.UserComment: b"UNICODE\0" + comment}}
    return piexif.dump(exif_dict)

# ── Tensor -> PIL (sin cambios) ──────────────────────────────────────────────
def tensor_to_pil(tensor):
    if hasattr(tensor, 'cpu'):
        arr = tensor.cpu().numpy()
    else:
        arr = np.array(tensor)

    if len(arr.shape) == 4:
        arr = arr[0]
    if len(arr.shape) == 3 and arr.shape[0] in [1, 3, 4, 12] and arr.shape[2] not in [1, 3, 4, 12]:
        arr = np.transpose(arr, (1, 2, 0))

    if len(arr.shape) == 3 and arr.shape[2] > 4:
        if arr.min() >= 0 and arr.max() <= 1:
            arr = arr[:, :, :3]
        else:
            rgb3 = arr[:, :, :3]
            arr = (rgb3 - rgb3.min()) / (rgb3.max() - rgb3.min() + 1e-8)

    if len(arr.shape) == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        arr = np.concatenate([arr, arr, arr], axis=2)
    if len(arr.shape) == 2:
        arr = np.stack([arr, arr, arr], axis=2)

    if arr.dtype in (np.float32, np.float64):
        if arr.max() > 1.0 or arr.min() < 0:
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = (arr / 65535 * 255).astype(np.uint8)
    elif arr.dtype in (np.int16, np.int32):
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)

    if len(arr.shape) != 3 or arr.shape[2] not in [1, 3, 4]:
        raise ValueError(f"Forma final inválida: {arr.shape}")

    if arr.shape[2] == 3: return Image.fromarray(arr, 'RGB')
    if arr.shape[2] == 4: return Image.fromarray(arr, 'RGBA')
    return Image.fromarray(arr[:, :, 0], 'L')

# ── Node ───────────────────────────────────────────────────────────────────
class K3NKSaveCompress:

    COMP_MODES = ["none", "optimize (PIL)", "lossless (oxipng)", "lossy+lossless (pngquant+oxipng)"]
    SPEEDS     = ["1 (slowest/best)", "3 (balanced)", "6 (fast)", "10 (fastest)"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images":           ("IMAGE",),
                "filename_prefix":  ("STRING", {
                    "default": "k3nk/image",
                    "tooltip": "📝 Prefix for saved filenames. Use %date:yyyy-MM-dd% for date."
                }),
                "format":           (["PNG", "WEBP", "JPEG"], {
                    "tooltip": "📁 Output format:\n• PNG: Best quality + compression\n• WEBP: Good for web\n• JPEG: Maximum compression (lossy)"
                }),
                "compression_mode": (cls.COMP_MODES, {
                    "tooltip": "🎚️ Compression method:\n• None: No compression (original size)\n• Lossless (oxipng): 100% color quality, ~30-70% smaller\n• Smart (pngquant+oxipng): Up to 90% smaller, color may change slightly"
                }),
                "quality_min":      ("INT", {
                    "default": 85, "min": 0, "max": 100,
                    "tooltip": "🎨 PNGQUANT min quality (0-100)\n• Higher = better color, less compression\n• 85-100: Minimal color loss (recommended)\n• 60-85: Noticeable color change\n• 0-60: Heavy color loss"
                }),
                "quality_max":      ("INT", {
                    "default": 99, "min": 0, "max": 100,
                    "tooltip": "🎨 PNGQUANT max quality (0-100)\n• Avoid 100 (causes pngquant exit 99 error)\n• 85-99: Minimal color change (recommended)\n• Below 85: Noticeable color change\n• Below 60: Heavy color loss"
                }),
                "pngquant_speed":   (cls.SPEEDS, {
                    "tooltip": "⚡ PNGQUANT speed (1-10)\n• 1: Slowest, best compression\n• 3: Balanced (recommended)\n• 6: Fast\n• 10: Fastest, slightly less compression"
                }),
                "oxipng_level":     ("INT", {
                    "default": 2, "min": 1, "max": 6,
                    "tooltip": "🔧 OXIPNG compression level (1-6)\n• 1: Fastest, ~30% size reduction\n• 2: Faster, ~50% size reduction (recommended)\n• 3: Balanced, ~65% size reduction\n• 4-6: Slowest, ~70-80% size reduction\n• Lossless: Colors 100% preserved"
                }),
                "lossy_quality":    ("INT", {
                    "default": 90, "min": 1, "max": 100,
                    "tooltip": "📸 WEBP/JPEG quality (1-100)\n• 90-100: Near-perfect quality\n• 70-90: Good quality (recommended)\n• 50-70: Noticeable artifacts\n• Below 50: Heavy artifacts\n• Always lossy: Colors may change"
                }),
            },
            "optional": {
                "positive_text": ("STRING", {"default": "", "multiline": True, "forceInput": True,
                    "tooltip": "🟢 Connect your positive prompt (CLIPTextEncode) here so Civitai auto-detects it"}),
                "negative_text": ("STRING", {"default": "", "multiline": True, "forceInput": True,
                    "tooltip": "🔴 Connect your negative prompt here so Civitai auto-detects it"}),
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
             positive_text="", negative_text="",
             prompt=None, extra_pnginfo=None):

        import folder_paths
        from datetime import datetime

        now = datetime.now()
        filename_prefix = re.sub(r'%date:([^%]+)%',
            lambda m: now.strftime(m.group(1)
                .replace("yyyy", "%Y").replace("MM", "%m").replace("dd", "%d")
                .replace("hh", "%H").replace("mm", "%M").replace("ss", "%S")),
            filename_prefix)

        speed_val = int(pngquant_speed.split(" ")[0])
        ext_map   = {"PNG": ".png", "WEBP": ".webp", "JPEG": ".jpg"}
        ext       = ext_map[format]
        saved     = []

        full_output_folder, filename, counter, subfolder, filename_prefix_out = \
            folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory(), 512, 512)

        if not isinstance(images, (list, tuple)):
            images = [images]

        a1111_params = _build_a1111_string(prompt, positive_text, negative_text) if prompt else None

        for idx, img_tensor in enumerate(images):
            try:
                pil = tensor_to_pil(img_tensor)

                pnginfo = PngInfo()
                if extra_pnginfo:
                    for k, v in extra_pnginfo.items():
                        pnginfo.add_text(k, json.dumps(v) if not isinstance(v, str) else v)
                if prompt:
                    pnginfo.add_text("prompt", json.dumps(prompt))
                if a1111_params:
                    pnginfo.add_text("parameters", a1111_params)

                out_path = Path(full_output_folder) / f"{filename}_{counter + idx:05}_{ext}"

                if format == "PNG":
                    buf = io.BytesIO()
                    pil.save(buf, format="PNG", pnginfo=pnginfo, optimize=False, compress_level=1)
                    out_path.write_bytes(buf.getvalue())
                    if compression_mode != "none":
                        _compress_queue.put((out_path, compression_mode, quality_min, quality_max, speed_val, oxipng_level))

                elif format == "WEBP":
                    if pil.mode == "RGBA": pil = pil.convert("RGB")
                    exif = _exif_bytes(a1111_params) if a1111_params else None
                    kwargs = {"quality": lossy_quality, "method": 6}
                    if exif: kwargs["exif"] = exif
                    pil.save(str(out_path), format="WEBP", **kwargs)

                elif format == "JPEG":
                    if pil.mode in ["RGBA", "P"]: pil = pil.convert("RGB")
                    exif = _exif_bytes(a1111_params) if a1111_params else None
                    kwargs = {"quality": lossy_quality, "optimize": True}
                    if exif: kwargs["exif"] = exif
                    pil.save(str(out_path), format="JPEG", **kwargs)

                saved.append({"filename": out_path.name, "subfolder": subfolder, "type": "output"})
                print(f"[K3NK Save] ✔ {out_path.name}")

            except Exception as e:
                print(f"[K3NK Save] ✘ Error procesando imagen {idx}: {e}")
                continue

        return {"ui": {"images": saved}}


NODE_CLASS_MAPPINGS        = {"K3NKSaveCompress": K3NKSaveCompress}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKSaveCompress": "K3NK Save & Compress 🗜️"}