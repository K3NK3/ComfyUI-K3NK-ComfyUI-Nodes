"""
K3NK Save & Compress - ULTRA ROBUSTO V3
Manejo CORRECTO de tensores con múltiples canales
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
GITHUB_TAG     = "binaries"

_IS_WIN   = platform.system() == "Windows"
_BINARIES = {
    "pngquant": "pngquant.exe" if _IS_WIN else "pngquant",
    "oxipng":   "oxipng.exe"   if _IS_WIN else "oxipng",
}

def _download_binary(name: str, filename: str) -> str:
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

# ── FUNCIÓN MEJORADA PARA NORMALIZAR TENSORES CON MÚLTIPLES CANALES ──────────
def tensor_to_pil(tensor):
    """
    Convierte CUALQUIER tensor de imagen a PIL Image.
    AHORA con manejo CORRECTO de tensores con múltiples canales (>3)
    """
    # Convertir a numpy si es tensor
    if hasattr(tensor, 'cpu'):
        arr = tensor.cpu().numpy()
    else:
        arr = np.array(tensor)
    
    print(f"[DEBUG] Forma original del tensor: {arr.shape}")
    print(f"[DEBUG] Tipo de dato: {arr.dtype}")
    print(f"[DEBUG] Rango de valores: {arr.min():.4f} a {arr.max():.4f}")
    
    # Caso 1: Si es 4D (batch, height, width, channels)
    if len(arr.shape) == 4:
        # Quitar batch
        arr = arr[0]
        print(f"[DEBUG] Después de quitar batch: {arr.shape}")
    
    # Caso 2: Si es 4D pero en formato (batch, channels, height, width)
    if len(arr.shape) == 4 and arr.shape[0] in [1, 3, 4, 12]:
        # Intentar detectar formato
        if arr.shape[3] not in [1, 3, 4, 12]:  # Si el último no es canal
            arr = arr[0]  # Quitar batch
            if arr.shape[0] in [1, 3, 4, 12]:
                arr = np.transpose(arr, (1, 2, 0))
                print(f"[DEBUG] Transpuesto (channels primero): {arr.shape}")
    
    # Caso 3: Si es 3D y los canales están primero
    if len(arr.shape) == 3 and arr.shape[0] in [1, 3, 4, 12]:
        if arr.shape[2] not in [1, 3, 4, 12]:
            arr = np.transpose(arr, (1, 2, 0))
            print(f"[DEBUG] Transpuesto (channels primero): {arr.shape}")
    
    # ─── MANEJO DE MÚLTIPLES CANALES ───
    # Si tenemos más de 4 canales, probablemente son latents o features
    if len(arr.shape) == 3 and arr.shape[2] > 4:
        print(f"[WARNING] ¡El tensor tiene {arr.shape[2]} canales!")
        print(f"[INFO] Intentando convertir a RGB usando diferentes métodos...")
        
        # Método 1: Tomar los primeros 3 canales como RGB
        rgb1 = arr[:, :, :3]
        
        # Método 2: Tomar canales específicos (asumiendo que los últimos 3 son RGB)
        rgb2 = arr[:, :, -3:]
        
        # Método 3: Promediar todos los canales para escala de grises
        gray = np.mean(arr, axis=2)
        gray = np.stack([gray, gray, gray], axis=2)
        
        # Método 4: Normalizar y usar los primeros 3 canales
        rgb3 = arr[:, :, :3]
        rgb3 = (rgb3 - rgb3.min()) / (rgb3.max() - rgb3.min() + 1e-8)
        
        # Decidir cuál usar (probamos diferentes estrategias)
        # Si los valores están en [0,1] usar rgb1 directamente
        if arr.min() >= 0 and arr.max() <= 1:
            print(f"[INFO] Usando primeros 3 canales como RGB (valores normalizados 0-1)")
            arr = rgb1
        else:
            # Si los valores no están normalizados, usar el método 4
            print(f"[INFO] Usando primeros 3 canales normalizados")
            arr = rgb3
        
        print(f"[DEBUG] Canales reducidos a: {arr.shape}")
    
    # Si tiene 4 canales (RGBA), convertir a RGB
    if len(arr.shape) == 3 and arr.shape[2] == 4:
        print(f"[INFO] Tensor RGBA, convirtiendo a RGB")
        # Si es float, asumir que alpha es el último canal
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = arr[:, :, :3]  # Quitar alpha
        else:
            # Si es uint8, quitar alpha
            arr = arr[:, :, :3]
    
    # Si tiene 3 canales, asumir RGB
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        print(f"[INFO] Tensor RGB detectado")
    
    # Si tiene 1 canal (escala de grises)
    if len(arr.shape) == 3 and arr.shape[2] == 1:
        print(f"[INFO] Tensor escala de grises, convirtiendo a RGB")
        arr = np.concatenate([arr, arr, arr], axis=2)
    
    # Si es 2D (height, width) - escala de grises
    if len(arr.shape) == 2:
        print(f"[INFO] Tensor 2D (escala de grises), convirtiendo a RGB")
        arr = np.stack([arr, arr, arr], axis=2)
    
    # Normalizar valores a 0-255
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() > 1.0 or arr.min() < 0:
            # Si no está en [0,1], normalizar
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).clip(0, 255)
        arr = arr.astype(np.uint8)
    elif arr.dtype == np.uint16:
        arr = (arr / 65535 * 255).astype(np.uint8)
    elif arr.dtype == np.int16 or arr.dtype == np.int32:
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    elif arr.dtype == np.uint8:
        pass  # Ya está en el formato correcto
    else:
        # Cualquier otro tipo, normalizar
        arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
    
    # Verificar dimensiones finales
    if len(arr.shape) != 3 or arr.shape[2] not in [1, 3, 4]:
        raise ValueError(f"Forma final inválida: {arr.shape}. Esperaba (height, width, 3) o (height, width, 4)")
    
    # Crear imagen PIL
    if arr.shape[2] == 3:
        return Image.fromarray(arr, 'RGB')
    elif arr.shape[2] == 4:
        return Image.fromarray(arr, 'RGBA')
    else:
        return Image.fromarray(arr[:, :, 0], 'L')

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
        from datetime import datetime

        # Expande variables %date:formato%
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

        # Si images no es una lista, convertirla en una
        if not isinstance(images, (list, tuple)):
            images = [images]

        for idx, img_tensor in enumerate(images):
            try:
                # Convertir el tensor a PIL usando nuestra función mágica
                pil = tensor_to_pil(img_tensor)
                print(f"[DEBUG] Imagen convertida: {pil.size}, modo: {pil.mode}")
                
                # Build PngInfo con metadatos
                pnginfo = PngInfo()
                if extra_pnginfo:
                    for k, v in extra_pnginfo.items():
                        pnginfo.add_text(k, json.dumps(v) if not isinstance(v, str) else v)
                if prompt:
                    pnginfo.add_text("prompt", json.dumps(prompt))

                out_path = Path(full_output_folder) / f"{filename}_{counter + idx:05}_{ext}"

                if format == "PNG":
                    # Guardar a buffer para manipular chunks
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
                    # Para WEBP convertir a RGB si es RGBA
                    if pil.mode == "RGBA":
                        pil = pil.convert("RGB")
                    pil.save(str(out_path), format="WEBP", quality=lossy_quality, method=6)

                elif format == "JPEG":
                    # Para JPEG convertir a RGB
                    if pil.mode in ["RGBA", "P"]:
                        pil = pil.convert("RGB")
                    pil.save(str(out_path), format="JPEG", quality=lossy_quality, optimize=True)

                saved.append({"filename": out_path.name, "subfolder": subfolder, "type": "output"})
                size_kb = out_path.stat().st_size // 1024
                print(f"[K3NK Save] ✔ {out_path.name}  ({size_kb} KB)")

            except Exception as e:
                print(f"[K3NK Save] ✘ Error procesando imagen {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return {"ui": {"images": saved}}


NODE_CLASS_MAPPINGS        = {"K3NKSaveCompress": K3NKSaveCompress}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKSaveCompress": "K3NK Save & Compress 🗜️"}
