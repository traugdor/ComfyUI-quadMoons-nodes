import numpy
import torch
import torchvision
import torchvision.transforms.functional as TF
import math

import os
import sys
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "comfy"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

import comfy.samplers
import latent_preview

class qmChangeBackground:
    @staticmethod
    def get_mask_from_segs(segs):
        ### Make mask from segs
        ### Shamelessly borrowed from Impact Pack
        shape = segs[0]
        h = shape[0]
        w = shape[1]

        mask = numpy.zeros((h, w), dtype=numpy.uint8)

        # combine all segs into a single mask
        for seg in segs[1]:
            cropped_mask = seg.cropped_mask
            crop_region = seg.crop_region
            mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(numpy.uint8)

        the_mask = torch.from_numpy(mask.astype(numpy.float32) / 255.0)

        if len(the_mask.shape) == 4:
            return the_mask.squeeze(0)

        elif len(the_mask.shape) == 2:
            return the_mask.unsqueeze(0)
        
        return the_mask

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": ("MODEL",),
                "model_2": ("MODEL",),
                "positive": ("CONDITIONING",),
                "positive_2": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "negative_2": ("CONDITIONING",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "segs_from_SEGM_Detector": ("SEGS",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "steps_2": ("INT", {"default": 12, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "denoise_2": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }
        return inputs
    
    CATEGORY = "QuadmoonNodes/sampling"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "qmChangeBackground"
    DESCRIPTION="This node is designed to work with Impact Pack from ltdrdata (and even borrows some code to ensure full compatibility with the segs output from the SEGM Detector). What it does is take an input image, and change the background using whatever process you determine as defined by the SEGS input. This 2-pass sampler will change the background in one pass and then resample the entire image as a whole to ensure the new background and the foreground are seamlessly blended. "

    def qmChangeBackground(self, model, model_2, positive, positive_2, negative, negative_2, image, vae, segs_from_SEGM_Detector, seed, steps, steps_2, cfg, sampler_name, scheduler, denoise, denoise_2):
        mask = self.get_mask_from_segs(segs_from_SEGM_Detector)
        
        ## We were going to blur the mask, but it kept throwing an error, just resample the whole image with a lower denoising after initial pass.

        ## invert the mask for inpainting
        mask = (1.-mask)

        ## apply mask to vae encode to prepare for sampling
        x = (image.shape[1] // vae.downscale_ratio) * vae.downscale_ratio
        y = (image.shape[2] // vae.downscale_ratio) * vae.downscale_ratio
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(image.shape[1], image.shape[2]), mode="bilinear")

        image = image.clone()
        if image.shape[1] != x or image.shape[2] != y:
            x_offset = (image.shape[1] % vae.downscale_ratio) // 2
            y_offset = (image.shape[2] % vae.downscale_ratio) // 2
            image = image[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        #grow mask by a few pixels to keep things seamless in latent space
        grow_by = 6
        kernel_tensor = torch.ones((1, 1, grow_by, grow_by))
        padding = math.ceil((grow_by - 1) / 2)

        mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask.round(), kernel_tensor, padding=padding), 0, 1)

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            image[:,:,:,i] -= 0.5
            image[:,:,:,i] *= m
            image[:,:,:,i] += 0.5

        ## define latent and populate with samples and noise mask
        latent = {}
        latent["samples"] = vae.encode(image)
        latent["noise_mask"] = mask_erosion[:,:,:x,:y].round()

        ## prepare to sample new image:
        disable_noise = False
        start_step = None
        last_step=None
        force_full_denoise=False
        qmlatent_image = latent["samples"]
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(qmlatent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, qmlatent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        ## second pass 12 steps, 0.56 denoise, noise mask removed.
        samples = comfy.sample.sample(model_2, noise, steps_2, cfg, sampler_name, scheduler, positive_2, negative_2, samples,
                                    denoise=denoise_2, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=None, callback=callback, disable_pbar=disable_pbar, seed=seed)
        out = latent.copy()
        out["samples"] = samples
        
        ## Decode
        out_image = vae.decode(out["samples"])

        return (out_image,)
    
