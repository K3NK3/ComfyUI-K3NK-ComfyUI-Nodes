import os
import re
import folder_paths
import torch
from safetensors.torch import save_file


class SaveLatentAbsolutePath:
    CATEGORY = "K3NK/latent"
    FUNCTION = "save"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": ""}),
            }
        }

    def save(self, samples, filename_prefix):
        output_dir = folder_paths.get_output_directory()

        # Separar directorio y prefijo del filename
        prefix_path = os.path.join(output_dir, filename_prefix)
        directory = os.path.dirname(prefix_path)
        prefix = os.path.basename(prefix_path)

        os.makedirs(directory, exist_ok=True)

        # Buscar el siguiente número disponible
        counter = 0
        while True:
            filename = f"{prefix}_{counter:05d}_.latent"
            full_path = os.path.join(directory, filename)
            if not os.path.exists(full_path):
                break
            counter += 1

        tensors = {}
        metadata = {}
        for k, v in samples.items():
            if isinstance(v, torch.Tensor):
                tensors[k] = v.clone()
            else:
                metadata[k] = str(v)

        save_file(tensors, full_path, metadata=metadata)
        return (samples,)


NODE_CLASS_MAPPINGS = {
    "SaveLatentAbsolutePath": SaveLatentAbsolutePath,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveLatentAbsolutePath": "Save Latent (Pass-Through)",
}