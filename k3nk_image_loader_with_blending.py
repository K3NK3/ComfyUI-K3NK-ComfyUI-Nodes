import os
import glob
import re
from PIL import Image, ImageOps
import torch
import numpy as np
import time

class K3NKImageLoaderWithBlending:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "placeholder": "Directory path"}),
                "auto_detect_sequences": ("BOOLEAN", {"default": False}),
                "sequence_frames": ("INT", {"default": 81, "min": 0, "max": 10000}),
                "overlap_frames": ("INT", {"default": 5, "min": 1, "max": 20}),
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
        if numbers:
            return int(numbers[-1])
        return 0
    
    def smootherstep_blend(self, frame1, frame2, alpha):
        smooth_alpha = alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0)
        return frame1 * (1.0 - smooth_alpha) + frame2 * smooth_alpha
        
    def load_and_blend_images(self, directory_path, auto_detect_sequences=False,
                              sequence_frames=81, overlap_frames=5, file_pattern="*.png"):
        
        # Auto-detectar sequence_frames basado en número de archivos .latent
        if auto_detect_sequences:
            # Contar archivos .latent
            latent_files = glob.glob(os.path.join(directory_path, "*.latent"))
            num_latents = len(latent_files)
            
            if num_latents > 0:
                # Contar PNGs
                png_files = glob.glob(os.path.join(directory_path, file_pattern))
                num_pngs = len(png_files)
                
                if num_pngs > 0:
                    sequence_frames = num_pngs // num_latents
                    print(f"🔍 Auto-detected: {num_pngs} PNGs ÷ {num_latents} .latent files = {sequence_frames} frames per sequence")
                else:
                    print(f"⚠️ No PNG files found, using manual sequence_frames: {sequence_frames}")
            else:
                print(f"⚠️ No .latent files found, using manual sequence_frames: {sequence_frames}")
        
        search_pattern = os.path.join(directory_path, file_pattern)
        all_files = glob.glob(search_pattern)
        
        if len(all_files) == 0:
            raise ValueError(f"No files found in '{directory_path}'")
        
        files_with_numbers = [(f, self.extract_number_from_filename(os.path.basename(f))) for f in all_files]
        files_with_numbers.sort(key=lambda x: x[1])
        sorted_files = [f[0] for f in files_with_numbers]
        
        print(f"📁 Found {len(sorted_files)} images")
        print(f"🎯 {sequence_frames} frames per sequence")
        print(f"🔄 {overlap_frames} overlap frames to blend")
        
        # Cargar TODAS las imágenes
        all_tensors = []
        for filepath in sorted_files:
            try:
                img = Image.open(filepath)
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr)
                all_tensors.append(tensor)
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue
        
        if len(all_tensors) == 0:
            raise ValueError("No images loaded")
        
        total_frames = len(all_tensors)
        print(f"✅ Loaded {total_frames} images")
        
        # Si no hay suficientes frames para siquiera una secuencia, devolver tal cual
        if total_frames <= sequence_frames or overlap_frames == 0:
            print("⏭️  Not enough frames for blending")
            return (torch.stack(all_tensors, dim=0),)
        
        print(f"\n🔧 Processing sequences...")
        
        # Calcular secuencias completas y frames sobrantes
        num_complete_sequences = total_frames // sequence_frames
        remaining_frames = total_frames % sequence_frames
        
        print(f"  Complete sequences: {num_complete_sequences}")
        print(f"  Remaining frames: {remaining_frames}")
        
        result_tensors = []
        
        # Primera secuencia: añadir completa
        for i in range(sequence_frames):
            result_tensors.append(all_tensors[i].clone())
        
        print(f"  Added first sequence: frames 0-{sequence_frames-1}")
        
        # Procesar secuencias completas restantes
        for seq in range(1, num_complete_sequences):
            start_idx = seq * sequence_frames
            
            print(f"\n  Sequence {seq+1}:")
            print(f"    Start index: {start_idx}")
            
            # Blending con la secuencia anterior
            prev_end_start = len(result_tensors) - overlap_frames
            
            print(f"    Blending frames {prev_end_start}-{prev_end_start+overlap_frames-1} with {start_idx}-{start_idx+overlap_frames-1}")
            
            for i in range(overlap_frames):
                prev_frame = result_tensors[prev_end_start + i]
                new_frame = all_tensors[start_idx + i]
                alpha = (i + 1) / (overlap_frames + 1)
                result_tensors[prev_end_start + i] = self.smootherstep_blend(prev_frame, new_frame, alpha)
            
            # Añadir resto de frames de la secuencia
            for i in range(overlap_frames, sequence_frames):
                result_tensors.append(all_tensors[start_idx + i].clone())
            
            print(f"    Added frames {start_idx+overlap_frames}-{start_idx+sequence_frames-1}")
        
        # Procesar frames sobrantes si existen
        if remaining_frames > 0:
            start_idx = num_complete_sequences * sequence_frames
            
            print(f"\n  ⚠️  Remaining frames: {remaining_frames} (not a full sequence)")
            
            # Calcular cuántos frames podemos blendear
            actual_overlap = min(overlap_frames, remaining_frames)
            prev_end_start = len(result_tensors) - actual_overlap
            
            print(f"  🔄 Blending {actual_overlap} frames with remaining {remaining_frames} frames")
            print(f"    Blending frames {prev_end_start}-{prev_end_start+actual_overlap-1} with {start_idx}-{start_idx+actual_overlap-1}")
            
            # Blendear cada frame
            for i in range(actual_overlap):
                prev_frame = result_tensors[prev_end_start + i]
                new_frame = all_tensors[start_idx + i]
                alpha = (i + 1) / (actual_overlap + 1)
                result_tensors[prev_end_start + i] = self.smootherstep_blend(prev_frame, new_frame, alpha)
                print(f"      Frame {prev_end_start+i}: blended with frame {start_idx+i} (alpha={alpha:.2f})")
            
            # Añadir el RESTO de los frames sobrantes (sin los que ya blendearon)
            for i in range(actual_overlap, remaining_frames):
                frame_idx = start_idx + i
                result_tensors.append(all_tensors[frame_idx].clone())
            
            print(f"    Added {remaining_frames - actual_overlap} remaining frames")
        else:
            print("  No remaining frames after complete sequences")
        
        print(f"\n📊 Final: {len(result_tensors)} frames (original: {total_frames})")
        
        # DEBUG: Verificar si hubo blending real
        if len(result_tensors) == total_frames:
            print("🔍 Checking for actual blending...")
            changed_count = 0
            for i in range(min(10, total_frames)):
                if not torch.allclose(result_tensors[i], all_tensors[i], rtol=1e-4):
                    changed_count += 1
            print(f"   {changed_count}/10 first frames were modified")
        
        return (torch.stack(result_tensors, dim=0),)

NODE_CLASS_MAPPINGS = {"K3NKImageLoaderWithBlending": K3NKImageLoaderWithBlending}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKImageLoaderWithBlending": "K3NK Image Loader (Blending)"}
