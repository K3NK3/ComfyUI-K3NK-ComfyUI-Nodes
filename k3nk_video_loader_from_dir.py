import os
import re
import gc
import torch


class K3NKVideoLoaderFromDir:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "Ruta al directorio con videos"}),
                "sort_method": (["None", "Alphabetical (ASC)", "Alphabetical (DESC)",
                                 "Numerical (ASC)", "Numerical (DESC)",
                                 "Datetime (ASC)", "Datetime (DESC)"],),
            },
            "optional": {
                "start_index": ("INT", {"default": 0, "min": 0, "max": 99999, "step": 1,
                                        "tooltip": "Índice del video a cargar"}),
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1,
                                           "tooltip": "Máximo frames a extraer (0 = todos)"}),
                "load_always": ("BOOLEAN", {"default": False,
                                            "label_on": "Enabled", "label_off": "Disabled"}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "MASK", "INT", "VHS_VIDEOINFO")
    RETURN_NAMES  = ("images", "masks", "frame_count", "video_info")
    FUNCTION      = "load_video"
    CATEGORY      = "K3NK/loaders"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("load_always", False):
            return float("NaN")
        return hash(frozenset({k: str(v) for k, v in kwargs.items()}))

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".3gp"}

    def _sort(self, items, base_path, method):
        def num(s):
            m = re.search(r"\d+", s)
            return int(m.group()) if m else float("inf")
        fp = lambda x: os.path.join(base_path, x)
        if method == "Alphabetical (ASC)":  return sorted(items)
        if method == "Alphabetical (DESC)": return sorted(items, reverse=True)
        if method == "Numerical (ASC)":     return sorted(items, key=lambda x: num(os.path.splitext(x)[0]))
        if method == "Numerical (DESC)":    return sorted(items, key=lambda x: num(os.path.splitext(x)[0]), reverse=True)
        if method == "Datetime (ASC)":      return sorted(items, key=lambda x: os.path.getmtime(fp(x)))
        if method == "Datetime (DESC)":     return sorted(items, key=lambda x: os.path.getmtime(fp(x)), reverse=True)
        return items

    def _extract_frames_av(self, video_path, frame_load_cap=0):
        import av

        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            src_fps    = float(stream.average_rate) if stream.average_rate else 0.0
            src_frames = stream.frames  # puede ser 0 si el contenedor no lo reporta
            src_w      = stream.width
            src_h      = stream.height
            src_dur    = float(container.duration / 1_000_000) if container.duration else 0.0

            print(f"  🎬 {os.path.basename(video_path)} | {src_fps:.2f} fps | {src_w}x{src_h} | {src_dur:.2f}s")

            frames = []
            for i, frame in enumerate(container.decode(stream)):
                if frame_load_cap and i >= frame_load_cap:
                    break
                arr = frame.to_ndarray(format="rgb24")
                frames.append(torch.from_numpy(arr).float() / 255.0)
                if (i + 1) % 100 == 0:
                    print(f"     {i+1} frames extraídos…")

        if not frames:
            raise ValueError(f"Sin frames en: {video_path}")

        loaded_n = len(frames)
        imgs     = torch.stack(frames, dim=0)                                      # [N,H,W,3]
        masks    = torch.zeros((loaded_n, src_h, src_w), dtype=torch.float32)     # [N,H,W]

        # Dict compatible con VHS_VIDEOINFO — mismo esquema que usa VHS
        video_info = {
            "source_fps":         src_fps,
            "source_frame_count": src_frames if src_frames > 0 else loaded_n,
            "source_duration":    src_dur,
            "source_width":       src_w,
            "source_height":      src_h,
            "loaded_fps":         src_fps,
            "loaded_frame_count": loaded_n,
            "loaded_duration":    loaded_n / src_fps if src_fps > 0 else 0.0,
            "loaded_width":       src_w,
            "loaded_height":      src_h,
        }

        print(f"  ✅ {loaded_n} frames | {src_w}x{src_h} | {src_fps:.2f} fps")
        gc.collect()
        return imgs, masks, video_info

    def load_video(self, directory, sort_method="None",
                   start_index=0, frame_load_cap=0, load_always=False):

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directorio no encontrado: '{directory}'")

        video_files = sorted([
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and os.path.splitext(f)[1].lower() in self.VIDEO_EXTENSIONS
        ])

        if not video_files:
            raise FileNotFoundError(f"No hay videos en '{directory}'")

        video_files = self._sort(video_files, directory, sort_method)
        total = len(video_files)

        if start_index >= total:
            raise ValueError(f"start_index={start_index} fuera de rango (hay {total} videos)")

        chosen    = video_files[start_index]
        full_path = os.path.join(directory, chosen)
        print(f"📂 [{start_index + 1}/{total}] {chosen}")

        imgs, masks, video_info = self._extract_frames_av(full_path, frame_load_cap)
        return (imgs, masks, imgs.shape[0], video_info)


NODE_CLASS_MAPPINGS        = {"K3NKVideoLoaderFromDir": K3NKVideoLoaderFromDir}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKVideoLoaderFromDir": "Load Video Batch From Dir (K3NK)"}