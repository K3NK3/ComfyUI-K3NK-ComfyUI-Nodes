"""K3NK Save Video - compresión ffmpeg en background (no bloquea workflow) + Civitai metadata"""
import subprocess, threading, queue as _queue, shutil, json, re
from pathlib import Path

_compress_queue = _queue.Queue()

def _find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe: return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    for c in [here/"bin"/"ffmpeg.exe", here/"ffmpeg.exe"]:
        if c.exists(): return str(c)
    return ""

def _ensure_ffmpeg():
    dest = Path(__file__).resolve().parent / "bin" / "ffmpeg.exe"
    if dest.exists(): return str(dest)
    dest.parent.mkdir(exist_ok=True)
    url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
    try:
        import urllib.request, zipfile, io as _io
        print("[K3NK Video] downloading ffmpeg...")
        data = urllib.request.urlopen(url).read()
        z = zipfile.ZipFile(_io.BytesIO(data))
        name = next(n for n in z.namelist() if n.endswith("bin/ffmpeg.exe"))
        with z.open(name) as f, open(dest, "wb") as out:
            out.write(f.read())
        print(f"[K3NK Video] ✔ ffmpeg ready at {dest}")
        return str(dest)
    except Exception as e:
        print(f"[K3NK Video] ✘ could not download ffmpeg: {e}")
        return ""

def _iter_boxes(data, start, end):
    pos = start
    while pos + 8 <= end:
        size = int.from_bytes(data[pos:pos+4], "big")
        typ = data[pos+4:pos+8]
        if size == 0: size = end - pos
        yield typ, pos, pos + size
        pos += size

def _find_box(data, typ, start=0, end=None):
    if end is None: end = len(data)
    for t, s, e in _iter_boxes(data, start, end):
        if t == typ: return s, e
    return None

def _copy_udta(src: Path, dst: Path):
    try:
        orig = src.read_bytes()
        m = _find_box(orig, b"moov")
        if not m: return False
        u = _find_box(orig, b"udta", m[0]+8, m[1])
        if not u: return False
        udta_bytes = orig[u[0]:u[1]]

        new = bytearray(dst.read_bytes())
        m2 = _find_box(new, b"moov")
        if not m2: return False
        mdat = _find_box(new, b"mdat")
        if mdat and mdat[0] < m2[0]:
            u2 = _find_box(new, b"udta", m2[0]+8, m2[1])
            if u2:
                new[u2[0]:u2[1]] = udta_bytes
            else:
                new[m2[1]:m2[1]] = udta_bytes
            new_size = (m2[1] - m2[0]) + (len(udta_bytes) - ((u2[1]-u2[0]) if u2 else 0))
            new[m2[0]:m2[0]+4] = new_size.to_bytes(4, "big")
            dst.write_bytes(bytes(new))
            return True
        return False
    except Exception as e:
        print(f"[K3NK Video] metadata copy failed: {e}")
        return False

# ── Civitai metadata helpers (reused from k3nk_save_compress.py) ──────────────
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

def _escape_ffmpeg_metadata(value):
    """Escape special characters for FFmpeg metadata."""
    value = str(value)
    value = value.replace("\\", "\\\\")
    value = value.replace(";", "\\;")
    value = value.replace("#", "\\#")
    value = value.replace("=", "\\=")
    value = value.replace("\n", "\\\n")
    return value

def _write_metadata_file(prompt_json: str = "", workflow_json: str = "") -> Path:
    """Write metadata to a temporary FFMETADATA1 file."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write(";FFMETADATA1\n")
    if prompt_json:
        tmp.write(f"prompt={_escape_ffmpeg_metadata(prompt_json)}\n")
    if workflow_json:
        tmp.write(f"workflow={_escape_ffmpeg_metadata(workflow_json)}\n")
    tmp.close()
    return Path(tmp.name)

def _ffmpeg_embed_metadata(path: Path, params: str = "", workflow_json: str = ""):
    """Embed metadata in MP4 file using ffmpeg FFMETADATA1 format."""
    try:
        ffmpeg = _find_ffmpeg() or _ensure_ffmpeg()
        if not ffmpeg:
            print("[K3NK Video] ffmpeg not available for metadata embedding"); return False
        
        tmp_video = path.with_suffix(".tmp.mp4")
        
        # Write metadata to temp file
        meta_file = _write_metadata_file(prompt_json=params, workflow_json=workflow_json)
        
        cmd = [ffmpeg, "-y", "-i", str(path), "-i", str(meta_file),
               "-map", "0", "-map_metadata", "1",
               "-movflags", "use_metadata_tags",
               "-c", "copy", str(tmp_video)]
        r = subprocess.run(cmd, capture_output=True)
        
        meta_file.unlink()
        
        if r.returncode == 0 and tmp_video.exists():
            path.unlink()
            tmp_video.rename(path)
            return True
        else:
            print(f"[K3NK Video] ⚠️ ffmpeg metadata embed failed: {r.stderr.decode(errors='ignore')[-400:]}")
            if tmp_video.exists(): tmp_video.unlink()
            return False
    except Exception as e:
        print(f"[K3NK Video] ⚠️ metadata embed error: {e}")
        return False

def _ffmpeg_compress(path: Path, crf: int, preset: str, fmt: str = "mp4", params: str = "", workflow_json: str = "", delete_uncompressed: bool = True):
    try:
        ffmpeg = _find_ffmpeg() or _ensure_ffmpeg()
        if not ffmpeg:
            print("[K3NK Video] ffmpeg not available"); return
        size_before = path.stat().st_size
        tmp_video = path.with_suffix(f".tmp.{fmt}")
        
        # Choose codec based on format
        if fmt == "webm":
            # VP9 with CRF (0-63, lower is better quality)
            cmd = [ffmpeg, "-y", "-i", str(path),
                   "-map", "0", "-map_metadata", "0",
                   "-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0",
                   "-preset", preset, "-auto-alt-ref", "1",
                   "-c:a", "libopus"]
        else:
            # H.264 with CRF
            cmd = [ffmpeg, "-y", "-i", str(path),
                   "-map", "0", "-map_metadata", "0",
                   "-movflags", "use_metadata_tags",
                   "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-c:a", "copy"]
        
        cmd.append(str(tmp_video))
        
        r = subprocess.run(cmd, capture_output=True)
        
        if r.returncode == 0 and tmp_video.exists():
            size_after = tmp_video.stat().st_size
            if size_after < size_before:
                if delete_uncompressed:
                    final = path
                    path.unlink(missing_ok=True)
                    tmp_video.rename(final)
                else:
                    final = path.with_suffix(f".compressed.{fmt}")
                    tmp_video.rename(final)
                pct = 100 - (size_after * 100 // size_before)
                print(f"[K3NK Video] ✔ {path.name}: {size_before//1024} KB → {size_after//1024} KB (-{pct}%)")
            else:
                tmp_video.unlink()
                print(f"[K3NK Video] = {path.name}: no improvement, keeping original")
        else:
            print(f"[K3NK Video] ✘ ffmpeg error: {r.stderr.decode(errors='ignore')[-800:]}")
            if tmp_video.exists(): tmp_video.unlink()
    except Exception as e:
        print(f"[K3NK Video] ✘ compression error: {e}")

def _worker():
    while True:
        args = _compress_queue.get()
        try: _ffmpeg_compress(*args)
        finally: _compress_queue.task_done()

threading.Thread(target=_worker, daemon=True).start()


class K3NKSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename_prefix": ("STRING", {"default": "k3nk/video",
                    "tooltip": "Prefix. Supports %date:yyyy-MM-dd%"}),
                "format": (["mp4", "webm"], {"tooltip": "Output container"}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0,
                    "tooltip": "Frames per second (default: 30)"}),
                "compress_background": ("BOOLEAN", {"default": True,
                    "tooltip": "Re-encodes with ffmpeg after saving, without blocking the workflow"}),
                "delete_uncompressed": ("BOOLEAN", {"default": True,
                    "tooltip": "Deletes the uncompressed file once compression finishes"}),
                "crf": ("INT", {"default": 23, "min": 0, "max": 51,
                    "tooltip": "x264 quality: lower = better quality, bigger file (18-28 recommended)"}),
                "preset": (["ultrafast","fast","medium","slow","veryslow"],
                    {"default": "medium", "tooltip": "Speed vs compression of the re-encode"}),
            },
            "optional": {
                "video": ("VIDEO", {"forceInput": True,
                    "tooltip": "Input video to save (use this OR images+audio)"}),
                "images": ("IMAGE", {"forceInput": True,
                    "tooltip": "Input images to create video from (use with audio)"}),
                "audio": ("AUDIO", {"forceInput": True,
                    "tooltip": "Optional audio to add to video"}),
                "positive_text": ("STRING", {"default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Positive prompt, for Civitai metadata"}),
                "negative_text": ("STRING", {"default": "", "multiline": True, "forceInput": True,
                    "tooltip": "Negative prompt, for Civitai metadata"}),
                "civitai_mode": (["comment", "description", "both", "none"],
                    {"default": "comment", "tooltip": "How to embed prompt for Civitai: comment=MP4 comment field, description=MP4 description field, both=both fields, none=only workflow"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "K3NK/IO"

    def save(self, filename_prefix, format, fps, compress_background, delete_uncompressed, crf, preset,
              video=None, images=None, audio=None,
              positive_text="", negative_text="", civitai_mode="comment",
              prompt=None, extra_pnginfo=None):
        import folder_paths
        from datetime import datetime

        now = datetime.now()
        filename_prefix = re.sub(r'%date:([^%]+)%',
            lambda m: now.strftime(m.group(1).replace("yyyy","%Y").replace("MM","%m")
                .replace("dd","%d").replace("hh","%H").replace("mm","%M").replace("ss","%S")),
            filename_prefix)

        full_output_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())

        out_path = Path(full_output_folder) / f"{filename}_{counter:05}_.{format}"

        # Build metadata dict like core SaveVideo node + Civitai prompt
        saved_metadata = None
        if prompt or extra_pnginfo or positive_text or negative_text:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)

            # Use uppercase field names for WebM (matches VHS format that ComfyUI recognizes)
            if format == "webm":
                if prompt is not None:
                    metadata["PROMPT"] = json.dumps(prompt)
                    # Embed full workflow JSON from extra_pnginfo if available, otherwise use prompt
                    if extra_pnginfo and "workflow" in extra_pnginfo:
                        metadata["WORKFLOW"] = json.dumps(extra_pnginfo["workflow"])
                    else:
                        metadata["WORKFLOW"] = json.dumps(prompt)
                    # Remove lowercase workflow field to avoid duplication
                    if "workflow" in metadata:
                        del metadata["workflow"]
                if civitai_mode != "none" and (positive_text or negative_text or prompt):
                    civitai_prompt = _build_a1111_string(prompt or {}, positive_text, negative_text)
                    if civitai_mode == "comment":
                        metadata["COMMENT"] = civitai_prompt
                    elif civitai_mode == "description":
                        metadata["DESCRIPTION"] = civitai_prompt
                    elif civitai_mode == "both":
                        metadata["COMMENT"] = civitai_prompt
                        metadata["DESCRIPTION"] = civitai_prompt
            else:
                # MP4 uses lowercase field names (matches core SaveVideo node)
                if prompt is not None:
                    metadata["prompt"] = prompt
                if civitai_mode != "none" and (positive_text or negative_text or prompt):
                    civitai_prompt = _build_a1111_string(prompt or {}, positive_text, negative_text)
                    if civitai_mode == "comment":
                        metadata["comment"] = civitai_prompt
                    elif civitai_mode == "description":
                        metadata["description"] = civitai_prompt
                    elif civitai_mode == "both":
                        metadata["comment"] = civitai_prompt
                        metadata["description"] = civitai_prompt

            if len(metadata) > 0:
                saved_metadata = metadata

        # Save video with embedded metadata
        if video is not None:
            # Save existing video object (without metadata first)
            try:
                video.save_to(str(out_path), metadata=None)
            except TypeError:
                video.save_to(str(out_path))

            # Add audio using ffmpeg first (if provided)
            if audio is not None:
                ffmpeg = _find_ffmpeg() or _ensure_ffmpeg()
                if ffmpeg:
                    import numpy as np
                    import torch
                    
                    waveform = audio['waveform']
                    sample_rate = audio.get('sample_rate', 48000)
                    
                    # Convert torch tensor to numpy
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.cpu().numpy()
                    
                    # Convert list to numpy
                    if isinstance(waveform, list):
                        waveform = np.array(waveform)
                    
                    # Get number of channels
                    channels = 1
                    if waveform.ndim == 2:
                        channels = waveform.shape[0]
                    
                    # Transpose if needed (channels first -> samples first)
                    if waveform.ndim == 2 and waveform.shape[0] != channels:
                        waveform = waveform.T
                    
                    # Convert to float32 bytes for ffmpeg
                    audio_data = waveform.astype(np.float32).tobytes()
                    
                    tmp_video = out_path.with_suffix(".tmp." + format)
                    cmd = [ffmpeg, "-y", "-i", str(out_path),
                           "-ar", str(sample_rate), "-ac", str(channels),
                           "-f", "f32le", "-i", "-",
                           "-map", "0", "-map", "1:a",
                           "-c:v", "copy",
                           "-c:a", "aac" if format == "mp4" else "libopus",
                           "-shortest", str(tmp_video)]
                    
                    r = subprocess.run(cmd, input=audio_data, capture_output=True)
                    if r.returncode == 0 and tmp_video.exists():
                        out_path.unlink()
                        tmp_video.rename(out_path)
                    else:
                        print(f"[K3NK Video] ⚠️ Failed to add audio: {r.stderr.decode(errors='ignore')[-400:]}")
                        if tmp_video.exists(): tmp_video.unlink()

            # Add metadata using ffmpeg LAST (always in final file)
            if saved_metadata:
                ffmpeg = _find_ffmpeg() or _ensure_ffmpeg()
                if ffmpeg:
                    meta_file = _write_metadata_file(
                        prompt_json=saved_metadata.get("PROMPT", ""),
                        workflow_json=saved_metadata.get("WORKFLOW", "")
                    )
                    tmp_video = out_path.with_suffix(".tmp." + format)
                    cmd = [ffmpeg, "-y", "-i", str(out_path), "-i", str(meta_file),
                           "-map", "0", "-map_metadata", "1",
                           "-c", "copy", str(tmp_video)]
                    r = subprocess.run(cmd, capture_output=True)
                    meta_file.unlink()
                    if r.returncode == 0 and tmp_video.exists():
                        out_path.unlink()
                        tmp_video.rename(out_path)
                    else:
                        print(f"[K3NK Video] ⚠️ Failed to embed metadata: {r.stderr.decode(errors='ignore')[-400:]}")
                        if tmp_video.exists(): tmp_video.unlink()

            print(f"[K3NK Video] ✔ {out_path.name}")
        elif images is not None:
            # Create video from images
            import av
            container = av.open(str(out_path), mode='w')
            
            # Determine codec based on format
            from fractions import Fraction
            fps_fraction = Fraction(fps).limit_denominator(1000)
            
            if format == 'mp4':
                video_stream = container.add_stream('libx264', rate=fps_fraction)
                video_stream.pix_fmt = 'yuv420p'
            else:  # webm
                video_stream = container.add_stream('libvpx-vp9', rate=fps_fraction)
                video_stream.pix_fmt = 'yuv420p'
            
            # Write frames (no audio stream - added separately with ffmpeg like VHS)
            import numpy as np
            import torch
            from PIL import Image
            
            for i, img in enumerate(images):
                pil_img = None
                
                # Handle torch tensor (ComfyUI standard format)
                if isinstance(img, torch.Tensor):
                    # Convert to numpy: (H, W, C) -> PIL
                    img_np = img.cpu().numpy()
                    if img_np.dtype == np.float32 or img_np.dtype == np.float64:
                        img_np = (img_np * 255).astype(np.uint8)
                    elif img_np.max() > 1.0:
                        img_np = img_np.astype(np.uint8)
                    pil_img = Image.fromarray(img_np)
                
                # Handle numpy array
                elif isinstance(img, np.ndarray):
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img * 255).astype(np.uint8)
                    pil_img = Image.fromarray(img)
                
                # Handle PIL Image directly
                else:
                    pil_img = img
                
                if pil_img is not None:
                    frame = av.VideoFrame.from_image(pil_img)
                    for packet in video_stream.encode(frame):
                        container.mux(packet)
            
            # Flush video encoder
            for packet in video_stream.encode(None):
                container.mux(packet)
            
            container.close()
            
            # Add audio using ffmpeg first (if provided)
            if audio is not None:
                ffmpeg = _find_ffmpeg() or _ensure_ffmpeg()
                if ffmpeg:
                    import numpy as np
                    import torch
                    
                    waveform = audio['waveform']
                    sample_rate = audio.get('sample_rate', 48000)
                    
                    # Convert torch tensor to numpy
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.cpu().numpy()
                    
                    # Convert list to numpy
                    if isinstance(waveform, list):
                        waveform = np.array(waveform)
                    
                    # Get number of channels
                    channels = 1
                    if waveform.ndim == 2:
                        channels = waveform.shape[0]
                    
                    # Transpose if needed (channels first -> samples first)
                    if waveform.ndim == 2 and waveform.shape[0] != channels:
                        waveform = waveform.T
                    
                    # Convert to float32 bytes for ffmpeg
                    audio_data = waveform.astype(np.float32).tobytes()
                    
                    # Mux audio into video using ffmpeg
                    tmp_video = out_path.with_suffix(f".tmp.{format}")
                    cmd = [ffmpeg, "-y", "-i", str(out_path),
                           "-ar", str(sample_rate), "-ac", str(channels),
                           "-f", "f32le", "-i", "-",
                           "-c:v", "copy",
                           "-c:a", "aac" if format == "mp4" else "libopus",
                           "-shortest", str(tmp_video)]
                    
                    r = subprocess.run(cmd, input=audio_data, capture_output=True)
                    if r.returncode == 0 and tmp_video.exists():
                        out_path.unlink()
                        tmp_video.rename(out_path)
                    else:
                        print(f"[K3NK Video] ⚠️ Failed to add audio: {r.stderr.decode(errors='ignore')[-400:]}")
                        if tmp_video.exists(): tmp_video.unlink()

            # Add metadata using ffmpeg LAST (always in final file)
            if saved_metadata:
                ffmpeg = _find_ffmpeg() or _ensure_ffmpeg()
                if ffmpeg:
                    meta_file = _write_metadata_file(
                        prompt_json=saved_metadata.get("PROMPT", ""),
                        workflow_json=saved_metadata.get("WORKFLOW", "")
                    )
                    tmp_video = out_path.with_suffix(f".tmp.{format}")
                    cmd = [ffmpeg, "-y", "-i", str(out_path), "-i", str(meta_file),
                           "-map", "0", "-map_metadata", "1",
                           "-c", "copy", str(tmp_video)]
                    r = subprocess.run(cmd, capture_output=True)
                    meta_file.unlink()
                    if r.returncode == 0 and tmp_video.exists():
                        out_path.unlink()
                        tmp_video.rename(out_path)
                    else:
                        print(f"[K3NK Video] ⚠️ Failed to embed metadata: {r.stderr.decode(errors='ignore')[-400:]}")
                        if tmp_video.exists(): tmp_video.unlink()

        else:
            raise ValueError("Either video or images must be provided")

        print(f"[K3NK Video] ✔ {out_path.name}")

        if compress_background:
            # Compress in background (metadata already embedded)
            _compress_queue.put((out_path, crf, preset, format, "", "", delete_uncompressed))

        return {"ui": {"images": [{"filename": out_path.name, "subfolder": subfolder, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"K3NKSaveVideo": K3NKSaveVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKSaveVideo": "K3NK Save Video 🎞️"}
