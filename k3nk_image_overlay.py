import torch
import torch.nn.functional as F
import gc
import math
import comfy.utils


class K3NKImageOverlay:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image":     ("IMAGE",),
                "overlay_image":  ("IMAGE",),
                "overlay_resize": (["None", "Fit", "Resize by rescale_factor", "Resize to width & heigth"],),
                "resize_method":  (["nearest-exact", "bilinear", "area"],),
                "rescale_factor": ("FLOAT", {"default": 1,   "min": 0.01,   "max": 16.0,    "step": 0.01}),
                "width":          ("INT",   {"default": 512,  "min": 0,      "max": 48000,   "step": 1}),
                "height":         ("INT",   {"default": 512,  "min": 0,      "max": 48000,   "step": 1}),
                "x_offset":       ("INT",   {"default": 0,    "min": -48000, "max": 48000,   "step": 1}),
                "y_offset":       ("INT",   {"default": 0,    "min": -48000, "max": 48000,   "step": 1}),
                "rotation":       ("INT",   {"default": 0,    "min": -180,   "max": 180,     "step": 1}),
                "opacity":        ("FLOAT", {"default": 0,    "min": 0,      "max": 100,     "step": 0.1}),
            },
            "optional": {
                "optional_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay_image"
    CATEGORY = "K3NK/image"

    def apply_overlay_image(self, base_image, overlay_image, overlay_resize, resize_method,
                            rescale_factor, width, height, x_offset, y_offset,
                            rotation, opacity, optional_mask=None):

        # --- 1. Resize overlay (same logic as original) ---
        if overlay_resize != "None":
            ov_w = overlay_image.shape[2]
            ov_h = overlay_image.shape[1]

            if overlay_resize == "Fit":
                h_ratio = base_image.shape[1] / ov_h
                w_ratio = base_image.shape[2] / ov_w
                ratio = min(h_ratio, w_ratio)
                new_w = round(ov_w * ratio)
                new_h = round(ov_h * ratio)
            elif overlay_resize == "Resize by rescale_factor":
                new_w = int(ov_w * rescale_factor)
                new_h = int(ov_h * rescale_factor)
            else:  # "Resize to width & heigth"
                new_w, new_h = width, height

            # comfy.utils.common_upscale expects [B, C, H, W]
            samples = overlay_image.movedim(-1, 1)
            samples = comfy.utils.common_upscale(samples, new_w, new_h, resize_method, False)
            overlay_image = samples.movedim(1, -1)
            del samples

        Ho, Wo = overlay_image.shape[1], overlay_image.shape[2]

        # --- 2. Build alpha channel [Ho, Wo] float32 0-1 ---
        if optional_mask is not None:
            mask = optional_mask
            if mask.dim() == 4:
                mask = mask.squeeze(1)
            if mask.dim() == 3:
                mask = mask[0]  # take first, same as original (mask.resize uses first frame)
            # Resize mask to overlay size
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                 size=(Ho, Wo), mode="bilinear", align_corners=False).squeeze()
            # Original does ImageOps.invert → invert mask (1 - mask)
            alpha = 1.0 - mask.clamp(0, 1)
        else:
            alpha = torch.ones(Ho, Wo, dtype=torch.float32)

        # Apply opacity: original does x * (1 - opacity/100)
        alpha = alpha * (1.0 - opacity / 100.0)  # [Ho, Wo]

        # --- 3. Rotate overlay and alpha ---
        if rotation != 0:
            angle_rad = math.radians(-rotation)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Expand canvas like PIL expand=True
            # New size after rotation
            new_w_r = int(abs(Wo * cos_a) + abs(Ho * sin_a))
            new_h_r = int(abs(Wo * sin_a) + abs(Ho * cos_a))

            # Rotate overlay [1, Ho, Wo, 3] → [1, 3, Ho, Wo]
            ov = overlay_image[0:1].permute(0, 3, 1, 2)
            # Pad to new size first
            pad_w = (new_w_r - Wo) // 2
            pad_h = (new_h_r - Ho) // 2
            ov = F.pad(ov, (pad_w, pad_w, pad_h, pad_h))
            theta = torch.tensor([[cos_a, -sin_a, 0],
                                   [sin_a,  cos_a, 0]],
                                  dtype=torch.float32).unsqueeze(0)
            grid = F.affine_grid(theta, ov.shape, align_corners=False)
            ov = F.grid_sample(ov, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            overlay_image = ov.permute(0, 2, 3, 1)  # [1, nH, nW, 3]
            del ov

            # Rotate alpha the same way
            al = alpha.unsqueeze(0).unsqueeze(0)  # [1, 1, Ho, Wo]
            al = F.pad(al, (pad_w, pad_w, pad_h, pad_h))
            al = F.grid_sample(al, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            alpha = al.squeeze()  # [nH, nW]
            del al

            Ho, Wo = overlay_image.shape[1], overlay_image.shape[2]
            gc.collect()

        # alpha [Ho, Wo] → [Ho, Wo, 1]
        alpha = alpha.unsqueeze(-1)

        # --- 4. Composite frame by frame ---
        B, H, W, C = base_image.shape

        ox1, oy1 = x_offset, y_offset
        bx1 = max(0, ox1);     by1 = max(0, oy1)
        bx2 = min(W, ox1 + Wo); by2 = min(H, oy1 + Ho)

        if bx1 >= bx2 or by1 >= by2:
            print("K3NK Image Overlay: overlay outside base image, returning base unchanged")
            return (base_image,)

        lx1 = bx1 - ox1; ly1 = by1 - oy1
        lx2 = lx1 + (bx2 - bx1); ly2 = ly1 + (by2 - by1)

        ov_crop = overlay_image[0, ly1:ly2, lx1:lx2, :]    # [rH, rW, 3]
        al_crop = alpha[ly1:ly2, lx1:lx2, :]               # [rH, rW, 1]

        result = base_image.clone()
        for i in range(B):
            base_crop = result[i, by1:by2, bx1:bx2, :]
            result[i, by1:by2, bx1:bx2, :] = (
                base_crop * (1.0 - al_crop) + ov_crop * al_crop
            ).clamp(0, 1)
            del base_crop

        del ov_crop, al_crop, alpha, overlay_image
        gc.collect()

        return (result,)


NODE_CLASS_MAPPINGS = {"K3NKImageOverlay": K3NKImageOverlay}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKImageOverlay": "K3NK Image Overlay"}
