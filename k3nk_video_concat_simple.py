import os
import sys
import glob
import re
import subprocess
import tempfile
import time

class K3NKVideoConcatSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": ""}),
                "output_filename": ("STRING", {"default": "concatenated_video"}),
                "output_format": (["mp4", "avi", "webm", "mov", "mkv", "gif"], {"default": "mp4"}),
            },
            "optional": {
                "output_absolute_path": ("STRING", {"default": "", "placeholder": "Ruta absoluta (si vacío usa launch arg)"}),
                "reencode": ("BOOLEAN", {"default": True, "label": ["Re-encode (lento)", "Copy stream (rápido)"]})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "concatenate"
    CATEGORY = "K3NK/video-tools"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def get_output_directory(self):
        args = sys.argv
        for i, arg in enumerate(args):
            if arg == '--output-directory' and i + 1 < len(args):
                return os.path.abspath(args[i + 1])
            elif arg.startswith('--output-directory='):
                return os.path.abspath(arg.split('=', 1)[1])
        comfyui_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        return os.path.join(comfyui_dir, "output")

    def extract_number(self, filename):
        numbers = re.findall(r'\d+', filename)
        return int(numbers[-1]) if numbers else 0

    def concatenate(self, directory_path, output_filename, output_format, output_absolute_path="", reencode=True):
        all_files = glob.glob(os.path.join(directory_path, "*.mp4"))
        if not all_files:
            raise ValueError(f"No videos found in: {directory_path}")
        
        files_with_numbers = sorted(
            [(f, self.extract_number(os.path.basename(f))) for f in all_files],
            key=lambda x: x[1]
        )
        sorted_files = [f[0] for f in files_with_numbers]
        
        print(f"Found {len(sorted_files)} videos")
        
        list_file = os.path.join(tempfile.gettempdir(), f"video_list_{int(time.time())}.txt")
        with open(list_file, 'w') as f:
            for filepath in sorted_files:
                safe_path = filepath.replace('\\', '/')
                f.write(f"file '{safe_path}'\n")
        
        if output_absolute_path:
            output_dir = os.path.dirname(output_absolute_path) or os.getcwd()
            output_path = output_absolute_path if output_absolute_path.endswith(f".{output_format}") else f"{output_absolute_path}.{output_format}"
        else:
            output_dir = self.get_output_directory()
            output_path = os.path.join(output_dir, f"{output_filename}.{output_format}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if output_format == "gif":
            # GIF siempre necesita re-encode
            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-vf', 'fps=10', '-c:v', 'gif', output_path]
        elif not reencode:
            # Copy stream - instantáneo, pero el formato debe ser compatible
            if output_format in ["mp4", "avi", "mkv"]:
                cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', output_path]
                print(f"⚡ Copy stream a {output_format.upper()} (rápido)...")
            elif output_format == "webm":
                # WebM no acepta H.264+AAC, redirigir a MP4
                mp4_path = output_path.replace(f".{output_format}", ".mp4")
                cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', mp4_path]
                print(f"❌ WebM NO soporta copy stream (solo VP9+Opus)")
                print(f"   → Guardando como MP4 (rápido): {mp4_path}")
            elif output_format == "mov":
                cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', output_path]
                print(f"⚡ Copy stream a {output_format.upper()} (rápido)...")
            else:
                print(f"❌ {output_format.upper()} NO soporta copy stream")
                mp4_path = output_path.replace(f".{output_format}", ".mp4")
                cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c', 'copy', mp4_path]
                print(f"   → Guardando como MP4 (rápido): {mp4_path}")
        else:
            # Re-encode
            if output_format == "webm":
                vcodec, acodec = "libvpx-vp9", "libopus"
            else:
                vcodec, acodec = "libx264", "aac"
            cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, '-c:v', vcodec]
            if acodec:
                cmd.extend(['-c:a', acodec])
            cmd.append(output_path)
        
        print(f"{'Re-encode' if reencode and output_format != 'gif' else 'Copy stream'} to {output_format}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        os.remove(list_file)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error: {result.stderr}")
        
        print(f"Saved: {output_path}")
        return output_path


NODE_CLASS_MAPPINGS = {"K3NKVideoConcatSimple": K3NKVideoConcatSimple}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKVideoConcatSimple": "K3NK Video Concat Simple"}