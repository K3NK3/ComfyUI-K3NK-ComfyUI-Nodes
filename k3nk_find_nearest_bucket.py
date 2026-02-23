class K3NKFindNearestBucket:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_resolution": ("INT", {"default": 640, "min": 64, "max": 2048, "step": 16}),
            }
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("width", "height",)
    FUNCTION = "process"
    CATEGORY = "K3NK/utils"

    def process(self, image, base_resolution):
        from .diffusers_helper.bucket_tools import find_nearest_bucket
        H, W = image.shape[1], image.shape[2]
        new_height, new_width = find_nearest_bucket(H, W, resolution=base_resolution)
        return (new_width, new_height,)


NODE_CLASS_MAPPINGS = {"K3NKFindNearestBucket": K3NKFindNearestBucket}
NODE_DISPLAY_NAME_MAPPINGS = {"K3NKFindNearestBucket": "K3NK Find Nearest Bucket"}