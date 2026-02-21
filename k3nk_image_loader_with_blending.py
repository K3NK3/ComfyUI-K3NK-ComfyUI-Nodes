import os
import glob
import re
from PIL import Image, ImageOps
import torch
import numpy as np
import time
import gc


class K3NKImageLoaderWithBlending:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "placeholder": "Directory path"}),
                "auto_detect_sequences": ("BOOLEAN", {"default": True}),
                "enable_blending": ("BOOLEAN", {"default": True}),
                "overlap_frames": ("INT", {"default": 10, "min": 1, "max": 50}),
                "file_pattern": ("STRING", {"default": "*.png"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_and_blend_images"
    CATEGORY = "K3NK/loaders"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def extract_number_from_filename(self, filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0

    def load_image_as_tensor(self, filepath):
        """Carga una imagen y devuelve tensor float32 [H, W, 3]"""
        img = Image.open(filepath)
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        img.close()
        return torch.from_numpy(arr)

    def get_image_shape(self, filepath):
        """Lee solo dimensiones sin cargar píxeles"""
        with Image.open(filepath) as img:
            w, h = img.size
        return h, w

    def find_latent_positions_in_pngs(self, directory_path, png_numbers):
        latent_files = glob.glob(os.path.join(directory_path, "*.latent"))

        if not latent_files:
            print("⚠️ No .latent files found")
            return []

        latent_numbers = sorted(
            [(self.extract_number_from_filename(os.path.basename(f)), os.path.basename(f))
             for f in latent_files]
        )

        print(f"\n🔍 Found {len(latent_numbers)} .latent files:")
        latent_positions = []

        for latent_num, latent_name in latent_numbers:
            position = next((idx for idx, png_num in enumerate(png_numbers) if png_num > latent_num), None)
            if position is not None:
                latent_positions.append(position)
                print(f"  📌 {latent_name} (#{latent_num}) → PNG index {position}")
            else:
                print(f"  ⚠️ {latent_name} (#{latent_num}) → No PNG found after this latent")

        return latent_positions

    def smootherstep_blend(self, frame1, frame2, alpha):
        smooth_alpha = alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)
        return frame1 * (1.0 - smooth_alpha) + frame2 * smooth_alpha

    def load_and_blend_images(self, directory_path, auto_detect_sequences=True,
                              enable_blending=True, overlap_frames=10, file_pattern="*.png"):

        search_pattern = os.path.join(directory_path, file_pattern)
        all_files = glob.glob(search_pattern)

        if not all_files:
            raise ValueError(f"No files found in '{directory_path}' with pattern '{file_pattern}'")

        files_with_numbers = sorted(
            [(f, self.extract_number_from_filename(os.path.basename(f))) for f in all_files],
            key=lambda x: x[1]
        )
        sorted_files = [f[0] for f in files_with_numbers]
        png_numbers = [f[1] for f in files_with_numbers]
        total_frames = len(sorted_files)

        print(f"\n📁 Found {total_frames} images")

        # Leer dimensiones sin cargar toda la imagen
        h, w = self.get_image_shape(sorted_files[0])
        print(f"📐 Image size: {w}x{h}")

        # --- Sin blending: pre-alocar y cargar frame a frame ---
        if not enable_blending or not auto_detect_sequences:
            print("⚡ Blending disabled — loading into pre-allocated tensor...")
            output = torch.empty((total_frames, h, w, 3), dtype=torch.float32)
            for i, filepath in enumerate(sorted_files):
                try:
                    output[i] = self.load_image_as_tensor(filepath)
                except Exception as e:
                    print(f"⚠️ Error loading {filepath}: {e}")
                if (i + 1) % 50 == 0:
                    gc.collect()
                    print(f"  Loaded {i + 1}/{total_frames}")
            print(f"✅ Done: {total_frames} frames")
            return (output,)

        # --- Con blending ---
        print(f"🔄 Overlap frames: {overlap_frames}")

        latent_positions = self.find_latent_positions_in_pngs(directory_path, png_numbers)

        if not latent_positions:
            print("⚠️ No .latent positions detected — falling back to no blending")
            return self.load_and_blend_images(directory_path, auto_detect_sequences,
                                              False, overlap_frames, file_pattern)

        valid_positions = [pos for pos in latent_positions if pos >= overlap_frames]

        if not valid_positions:
            print("⚠️ No valid transition points — falling back to no blending")
            return self.load_and_blend_images(directory_path, auto_detect_sequences,
                                              False, overlap_frames, file_pattern)

        if len(valid_positions) < len(latent_positions):
            print(f"⚠️ Skipped {len(latent_positions) - len(valid_positions)} .latent files (too early)")

        latent_positions = valid_positions
        n_transitions = len(latent_positions)

        # Calcular límites de secuencias
        sequence_boundaries = []
        for i, pos in enumerate(latent_positions):
            if i == 0:
                sequence_boundaries.append((0, pos - 1))
            start = pos
            end = latent_positions[i + 1] - 1 if i + 1 < len(latent_positions) else total_frames - 1
            sequence_boundaries.append((start, end))

        # Frames de salida = total - frames solapados descartados
        output_frames = total_frames - n_transitions * overlap_frames

        print(f"\n📊 Detected {len(sequence_boundaries)} sequences:")
        for i, (start, end) in enumerate(sequence_boundaries):
            print(f"  Sequence {i + 1}: frames {start}–{end} ({end - start + 1} frames)")
        print(f"📦 Pre-allocating output: {output_frames} frames @ {w}x{h}")

        # Pre-alocar tensor de salida — escribimos directamente, nunca hay doble copia
        output = torch.empty((output_frames, h, w, 3), dtype=torch.float32)
        out_idx = 0  # cursor de escritura

        for seq_idx, (seq_start, seq_end) in enumerate(sequence_boundaries):
            seq_len = seq_end - seq_start + 1

            if seq_idx == 0:
                print(f"\n  Loading sequence 1: frames {seq_start}–{seq_end}")
                for i in range(seq_start, seq_end + 1):
                    output[out_idx] = self.load_image_as_tensor(sorted_files[i])
                    out_idx += 1
                gc.collect()
                continue

            actual_overlap = min(overlap_frames, seq_len)
            print(f"\n  Sequence {seq_idx + 1}: blending {actual_overlap} frames at boundary")

            # Sobreescribir in-place los últimos `actual_overlap` ya escritos
            blend_start = out_idx - actual_overlap
            for i in range(actual_overlap):
                new_frame = self.load_image_as_tensor(sorted_files[seq_start + i])
                alpha = (i + 1) / (actual_overlap + 1)
                output[blend_start + i] = self.smootherstep_blend(
                    output[blend_start + i], new_frame, alpha
                )
                del new_frame

            # Escribir el resto de la secuencia
            for i in range(actual_overlap, seq_len):
                output[out_idx] = self.load_image_as_tensor(sorted_files[seq_start + i])
                out_idx += 1

            gc.collect()

        print(f"\n📊 Final: {out_idx} frames written (original: {total_frames})")
        return (output,)


NODE_CLASS_MAPPINGS = {"K3NKImageLoaderWithBlending": K3NKImageLoaderWithBlending}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKImageLoaderWithBlending": "K3NK Image Loader (Blending)"}
