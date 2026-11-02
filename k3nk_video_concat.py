import os
import glob
import re
import torch
import numpy as np
import subprocess
import tempfile
import time

class K3NKVideoConcat:
    """
    Concatena videos .mp4 del directorio especificado.
    Devuelve IMAGE (tensor [T, H, W, C]) para usar con Video Combine.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "placeholder": "Directorio con videos .mp4"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video_frames",)
    FUNCTION = "concatenate"
    CATEGORY = "K3NK/video-tools"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def extract_number(self, filename):
        """Extrae el número del nombre de archivo para ordenar correctamente"""
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0

    def concatenate(self, directory_path):
        """
        Concatena todos los videos .mp4 del directorio.
        Devuelve tensor IMAGE [T, H, W, C].
        """
        
        # Buscar todos los archivos .mp4
        search_pattern = os.path.join(directory_path, "*.mp4")
        all_files = glob.glob(search_pattern)
        
        if not all_files:
            raise ValueError(f"No se encontraron videos .mp4 en: {directory_path}")
        
        # Ordenar por número en el nombre de archivo
        files_with_numbers = sorted(
            [(f, self.extract_number(os.path.basename(f))) for f in all_files],
            key=lambda x: x[1]
        )
        sorted_files = [f[0] for f in files_with_numbers]
        
        print(f"\n📁 Encontrados {len(sorted_files)} videos")
        for f in sorted_files:
            print(f"   - {os.path.basename(f)}")
        
        # Crear archivo de lista para ffmpeg concat
        list_file = os.path.join(tempfile.gettempdir(), f"video_list_{int(time.time())}.txt")
        with open(list_file, 'w') as f:
            for filepath in sorted_files:
                safe_path = filepath.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")
        
        # Concatenar videos con ffmpeg (incluye audio)
        output_path = os.path.join(tempfile.gettempdir(), f"concat_{int(time.time())}.mp4")
        
        cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', list_file,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ]
        
        print("🎬 Concatenando videos...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            os.remove(list_file)
            raise RuntimeError(f"FFmpeg concat error: {result.stderr[:500]}")
        
        os.remove(list_file)
        print(f"✅ Videos concatenados: {output_path}")
        
        # Extraer frames usando imageio directamente
        frames = self._extract_frames_with_imageio(output_path)
        
        # Limpiar archivo temporal
        os.remove(output_path)
        
        if len(frames) == 0:
            raise ValueError("No se pudieron extraer frames del video concatenado")
        
        # Stack frames y devolver tensor IMAGE [T, H, W, C]
        video_tensor = torch.stack(frames, dim=0)
        print(f"✅ Extraídos {len(frames)} frames - Shape: {video_tensor.shape}")
        
        return video_tensor

    def _extract_frames_with_imageio(self, video_path):
        """
        Extrae frames de un video MP4 usando imageio.
        Retorna lista de tensors PIL convertidos a torch.
        """
        frames = []
        
        print(f"   🎥 Abriendo video: {video_path}")
        
        try:
            import imageio
            
            # Usar imageio tradicional (v2 API)
            reader = imageio.get_reader(video_path, 'ffmpeg')
            
            for i, frame in enumerate(reader):
                if i == 0:
                    print(f"   📊 Video dims: {reader.get_size()}")
                
                # Convertir a RGB si es RGBA
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                elif len(frame.shape) == 2:
                    # Escala de grises a RGB
                    frame = np.stack([frame] * 3, axis=-1)
                
                # Convertir a tensor normalizado [0-1]
                frame_tensor = torch.from_numpy(np.array(frame).astype(np.float32) / 255.0)
                frames.append(frame_tensor)
                
                if (i + 1) % 100 == 0:
                    print(f"   📸 Progreso: {i + 1} frames...")
            
            print(f"   ✅ Video procesado correctamente")
            
        except Exception as e:
            print(f"   ❌ Error con imageio: {e}")
            raise RuntimeError(f"Error procesando video: {e}")
        
        return frames


NODE_CLASS_MAPPINGS = {
    "K3NKVideoConcat": K3NKVideoConcat
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K3NKVideoConcat": "K3NK Video Concat"
}
