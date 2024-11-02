import torch

import os
import sys
import safetensors.torch

import re
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "comfy"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

import comfy.samplers
import latent_preview

"""
    This code was shamelessly copied from the ComfyUI code itself. I am quite fortunate to have found it and
    I give all credit to the original authors for their work. I simply made a couple of modifications to it for everyone to enjoy.
"""

class qmKSampler:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "latent_image": ("LATENT", ),
                        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "upscale_latent": (["Yes", "No"],),
                    },
                    "optional":
                    {
                        "upscale_method": (s.upscale_methods,),
                        "ratio": ("FLOAT", {"default": 1.5, "min":0.01, "max":8.0, "step": 0.01}),
                    }
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING","INT","LATENT","LATENT")
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", "SEED", "LATENT","UPSCALED_LATENT")
    FUNCTION = "qmSample"

    CATEGORY = "QuadmoonNodes/sampling"

    def qmSample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, upscale_latent, upscale_method, ratio):
        disable_noise = False
        start_step = None
        last_step=None
        force_full_denoise=False
        latent = latent_image
        qmlatent_image = latent["samples"]
        if disable_noise:
            noise = torch.zeros(qmlatent_image.size(), dtype=qmlatent_image.dtype, layout=qmlatent_image.layout, device="cpu")
        else:
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
        out = latent.copy()
        out["samples"] = samples
        if(upscale_latent == "Yes"):
            s = latent.copy()
            width = round(out["samples"].shape[3] * ratio)
            height = round(out["samples"].shape[2] * ratio)
            s["samples"] = comfy.utils.common_upscale(out["samples"], width, height, upscale_method, "disabled")
            return (model, positive, negative, seed, out, s,)
        return (model, positive, negative, seed, out, out,)

class qmKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "add_noise": (["enable", "disable"], ),
                        "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "latent_image": ("LATENT", ),
                        "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                        "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                        "return_with_leftover_noise": (["disable", "enable"], ),
                    }
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING","INT","LATENT",)
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", "SEED", "LATENT",)
    FUNCTION = "qmSampleAdvanced"

    CATEGORY = "QuadmoonNodes/sampling"

    def qmSampleAdvanced(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
        latent = latent_image
        qmlatent_image = latent["samples"]
        if disable_noise:
            noise = torch.zeros(qmlatent_image.size(), dtype=qmlatent_image.dtype, layout=qmlatent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(qmlatent_image, noise_seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, qmlatent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)
        out = latent.copy()
        out["samples"] = samples
        return (model, positive, negative, noise_seed, out,)

class qmRotationalSampler:
    ### This attempts to mimic A1111 sampling of images with alternating prompts
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "", "multiline": True,}),
                "negPrompt": ("STRING", {"default": "", "multiline": True,}),
                "model": ("MODEL",),
                "clip": ("CLIP", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "image_advance": ("INT", {"default": 4, "min": 1, "max": 10}),
                "weight": ("FLOAT", {"default": 0.5, "min": 0.1, "max":1.0, "step":0.05, "round": 0.05}),
                "new_seed_after_steps": ("INT", {"default": 1, "min":1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "latent_image": ("LATENT", ),
                "upscale_latent": (["Yes", "No"],),
            },
            "optional":
            {
                "upscale_method": (s.upscale_methods,),
                "ratio": ("FLOAT", {"default": 1.5, "min":0.01, "max":8.0, "step": 0.01}),
            },
            "hidden": {
                
            }
        }
    
    RETURN_TYPES = ("LATENT","STRING", "STRING")
    RETURN_NAMES = ("LATENT","POSITIVE", "NEGATIVE")
    FUNCTION = "qmRotate"

    CATEGORY = "QuadmoonNodes/sampling"
    
    def qmRotate(self, prompt, negPrompt, model, clip, seed, steps, cfg, image_advance, weight, new_seed_after_steps, sampler_name, scheduler, latent_image, upscale_latent, upscale_method, ratio, denoise=1.0):
        word_dict = []
        pcounter = 0
        def replace_pattern(match):
            nonlocal pcounter
            keywords = match.group(1).split('|')
            word_dict.append(keywords)
            placeholder = f"qmRot{pcounter}"
            pcounter += 1
            return placeholder
        pattern = r"\[([^\[]+\|[^\]]+)\]"
        new_prompt = re.sub(pattern, replace_pattern, prompt)

        #recursively sample alternating each set of keywords
        disable_noise = False
        force_full_denoise=True
        latent = latent_image
        qmlatent_image = latent["samples"]
        if disable_noise:
            noise = torch.zeros(qmlatent_image.size(), dtype=qmlatent_image.dtype, layout=qmlatent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(qmlatent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        out = latent.copy()
        qmlatent_image = latent["samples"]
        if len(word_dict) > 0:
            for i in range (steps):
                if disable_noise:
                    noise = torch.zeros(qmlatent_image.size(), dtype=qmlatent_image.dtype, layout=qmlatent_image.layout, device="cpu")
                else:
                    batch_inds = latent["batch_index"] if "batch_index" in latent else None
                    noise = comfy.sample.prepare_noise(qmlatent_image, seed, batch_inds)

                noise_mask = None
                if "noise_mask" in latent:
                    noise_mask = latent["noise_mask"]
                tprompt = new_prompt
                dictpos = i % 2
                dictcount = 0
                for set in word_dict:
                    tprompt = tprompt.replace(f"qmRot{dictcount}", set[dictpos])
                    dictcount += 1
                tokens = clip.tokenize(tprompt)
                p, ppooled = clip.encode_from_tokens(tokens, return_pooled=True)
                positive = [[p, {"pooled_output": ppooled}]]
                tokens = clip.tokenize(negPrompt)
                n, npooled = clip.encode_from_tokens(tokens, return_pooled=True)
                negative = [[n, {"pooled_output": npooled}]]
                callback = latent_preview.prepare_callback(model, steps)
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
                if i < steps*weight: ## generating the image for a few steps can clean up the generation. Lower weights may produce a different output.
                    last_step = i+1+image_advance 
                else:
                    last_step = i+2
                if last_step > steps:
                    last_step = steps
                outsamples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, qmlatent_image,
                                        denoise=denoise, disable_noise=disable_noise, start_step=i+1, last_step=last_step,
                                        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
                qmlatent_image = outsamples
                if i % new_seed_after_steps == 0: ## using a new seed in the middle of generation can clean up the image if there are artifacts or the image is blurry
                    seed = random.randint(0, 0xFFFFFFFFFFFFFFFF)
        else:
            tokens = clip.tokenize(prompt)
            p, ppooled = clip.encode_from_tokens(tokens, return_pooled=True)
            positive = [[p, {"pooled_output": ppooled}]]
            tokens = clip.tokenize(negPrompt)
            n, npooled = clip.encode_from_tokens(tokens, return_pooled=True)
            negative = [[n, {"pooled_output": npooled}]]
            callback = latent_preview.prepare_callback(model, steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            outsamples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, qmlatent_image,
                                        denoise=denoise, disable_noise=disable_noise, start_step=None, last_step=None,
                                        force_full_denoise=False, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
            qmlatent_image = outsamples
        ##
        out["samples"] = qmlatent_image
        if(upscale_latent == "Yes"):
            s = latent.copy()
            width = round(out["samples"].shape[3] * ratio)
            height = round(out["samples"].shape[2] * ratio)
            s["samples"] = comfy.utils.common_upscale(out["samples"], width, height, upscale_method, "disabled")
            out = s
        return (out, prompt, negPrompt,)
    
class qmLatentImage:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    orientations = ["original", "force-landscape", "force-portrait"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "width": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": 16384, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "orientation": (s.orientations,)
                }
            }
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "qmLatent"

    CATEGORY = "latent"

    def qmLatent(self, width, height, batch_size, orientation):
        #if width > height it is already landscape
        match orientation:
            case "force-landscape":
                if (height > width): ## if it is portrait, then swap
                    temp = height
                    height = width
                    width = temp
            case "force-portrait":
                if (width > height): ## if it is landscape, then swap
                    temp = height
                    height = width
                    width = temp

        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )

class qmKSamplerBatched:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                        "model": ("MODEL",),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                        "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                        "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                        "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                        "positive": ("CONDITIONING", ),
                        "negative": ("CONDITIONING", ),
                        "latent_image": ("LATENT", {"tooltip": "A single latent or batch of latents that you wish to process individually."}),
                        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                    "optional":
                    {

                    }
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING","INT","LATENT",)
    RETURN_NAMES = ("MODEL", "POSITIVE", "NEGATIVE", "SEED", "LATENT",)
    FUNCTION = "qmSample"
    DESCRIPTION = "Allows for sampling of multiple latents when memory does not allow them all to be sampled at once."

    CATEGORY = "QuadmoonNodes/sampling"

    @staticmethod
    def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        disable_noise = False
        start_step = None
        last_step=None
        force_full_denoise=False
        latent = latent_image
        qmlatent_image = latent["samples"]
        print("Shape of latent", qmlatent_image.shape)
        if disable_noise:
            noise = torch.zeros(qmlatent_image.size(), dtype=qmlatent_image.dtype, layout=qmlatent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(qmlatent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        return comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, qmlatent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

    def qmSample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        individual_latents = torch.split(latent_image["samples"], 1, dim=0)  # Split along the batch dimension
        processed_latents = []

        # Loop over each latent and process it individually
        for i, single_latent in enumerate(individual_latents):
            # Modify the seed by adding an offset for each latent

            # Prepare the latent as a single-element batch dictionary for `common_ksampler`
            single_latent_dict = {"samples": single_latent, "batch_index": i}
            processed_latent = self.sample(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, single_latent_dict, denoise=denoise
            )
            
            # Append the processed latent to the list
            processed_latents.append(processed_latent)

        # Stack processed latents back into a batch
        latent_batch = torch.cat(processed_latents, dim=0)  # Combine along the batch dimension

        out = latent_image.copy()
        out["samples"] = latent_batch
        return (model, positive, negative, seed, out,)