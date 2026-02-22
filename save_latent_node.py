import os
import folder_paths
import torch
from safetensors.torch import save_file


class K3NKSaveLatentPassThrought:
    CATEGORY = "K3NK/latent"
    FUNCTION = "save"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    OUTPUT_NODE = True

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            }
        }

    def save(self, samples, filename_prefix):
        filename_prefix += self.prefix_append

        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, self.output_dir)

        file = f"{filename}_{counter:05d}_.latent"
        full_path = os.path.join(full_output_folder, file)

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
    "K3NKSaveLatentPassThrought": K3NKSaveLatentPassThrought,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "K3NKSaveLatentPassThrought": "K3NK Save Latent (Pass-Through)",
}
