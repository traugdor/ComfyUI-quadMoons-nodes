import torch

import os
import sys

# Assuming your script is two folders deep
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "comfy"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

import comfy

class qmBatchFromLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "latent": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."})
            }
        }
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The empty latent image batch.",)
    FUNCTION = "qmCopy"

    CATEGORY = "latent"
    DESCRIPTION = "Create a new batch of latent images from the input to be denoised via sampling."

    def qmCopy(self, latent, batch_size=1):
        latent_tensor = latent["samples"].to(self.device).view(1, *latent["samples"].shape[1:])
        latent_batch = latent_tensor.repeat(batch_size, 1, 1, 1)
        return ({"samples":latent_batch}, )