import math


def find_nearest_bucket(h, w, resolution=640):
    """
    Finds the nearest resolution bucket while preserving aspect ratio.
    Compatible with Wan 2.x, FramePack and any model using resolution buckets.

    Logic mirrors FramePack/diffusers_helper/bucket_tools.py:
    - Computes target pixel area from resolution
    - Searches for H and W multiples of 32 that preserve aspect ratio and are closest to target area
    """
    # Target pixel area
    target_pixels = resolution * resolution

    aspect = w / h

    # Buscar la mejor combinacion multiplo de 32
    best_h, best_w = resolution, resolution
    best_diff = float("inf")

    # Probar distintos valores de height multiplos de 32
    for test_h in range(32, resolution * 2 + 1, 32):
        test_w = round((test_h * aspect) / 32) * 32
        if test_w < 32:
            continue
        pixels = test_h * test_w
        diff = abs(pixels - target_pixels)
        if diff < best_diff:
            best_diff = diff
            best_h, best_w = test_h, test_w

    return best_h, best_w


class K3NKFindNearestBucket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_resolution": ("INT", {"default": 640, "min": 64, "max": 2048, "step": 32,
                                            "tooltip": "Base resolution. Use 640 for 480p, 720 for 720p, 1088 for Wan 2.x HD"}),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "process"
    CATEGORY = "K3NK/utils"

    def process(self, image, base_resolution):
        H, W = image.shape[1], image.shape[2]
        new_height, new_width = find_nearest_bucket(H, W, resolution=base_resolution)
        print(f"K3NK Find Nearest Bucket: input {W}x{H} → output {new_width}x{new_height} (base_resolution={base_resolution})")
        return (new_width, new_height,)


NODE_CLASS_MAPPINGS = {"K3NKFindNearestBucket": K3NKFindNearestBucket}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKFindNearestBucket": "K3NK Find Nearest Bucket"}
