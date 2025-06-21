from typing import TypeAlias

import os
import sys
import re
import math

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "comfy"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

comfy_root = os.path.abspath(os.path.join(__file__, "../../"))
sys.path.insert(0, comfy_root)

import comfy.utils
import folder_paths
import node_helpers

from PIL import Image
from PIL.PngImagePlugin import PngInfo

number: TypeAlias = int | float

class IntToString:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "integer_input": ("INT", {})
            }
        }
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "qmConvertIntToString"
    CATEGORY = "QuadmoonNodes/Converters"

    def qmConvertIntToString(self, integer_input):
        retval = ""
        if isinstance(integer_input, int):
            retval = (str(integer_input),)
        else:
            print("[WARN] - Quadmoon's Nodes Convert Integer to String requires an integer input. This will have undesired output");
            raise ValueError("Input is not an integer.")
        return retval


class FloatToString:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "float_input": ("FLOAT", {})
            }
        }
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "qmConvertFloatToString"
    CATEGORY = "QuadmoonNodes/Converters"

    def qmConvertFloatToString(self, float_input):
        retval = ""
        if isinstance(float_input, float):
            retval = (str(float_input),)
        else:
            print("[WARN] - Quadmoon's Nodes Convert Float to String requires a float input. This will have undesired output");
            raise ValueError("Input is not a float.")
        return retval

class NumberToString:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "number_input": ("NUMBER", {"ForceInput": True})
            }
        }
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "qmConvertIntToString"
    CATEGORY = "QuadmoonNodes/Converters"

    def qmConvertIntToString(self, number_input):
        retval = ""
        if isinstance(number_input, number):
            retval = (str(number_input),)
        else:
            print("[WARN] - Quadmoon's Nodes Convert Number to String requires a number input. This will have undesired output");
            raise ValueError("Input is not a number.")
        return retval

class BoolToString:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "boolean_input": ("BOOL", {"ForceInput": True})
            }
        }
        return inputs
    RETURN_TYPES = ("STRING",)
    FUNCTION = "qmConvertIntToString"
    CATEGORY = "QuadmoonNodes/Converters"

    def qmConvertIntToString(self, boolean_input):
        retval = ""
        if isinstance(boolean_input, bool):
            retval = (str(boolean_input),)
        else:
            print("[WARN] - Quadmoon's Nodes Convert Boolean to String requires a boolean input. This will have undesired output");
            raise ValueError("Input is not a boolean.")
        return retval
    
class NormalizeHW:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "image": ("IMAGE",),
                "platform":(["SD1.5", "XL", "FLUX"],),
                "orientation":(["original", "portrait", "landscape"],),
                "attention": (["minimize", "maximize"],)
            }
        }
        return inputs
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "qmNormalizeHW"
    CATEGORY = "QuadmoonNodes/Converters"

    def qmNormalizeHW(self, image,  platform, orientation, attention):
        divisor = 0.0
        height = image.shape[1]
        width = image.shape[2]
        newheight = 0
        newwidth = 0
        temp = 0

        match platform:
            case "SD1.5":
                newheight, newwidth, divisor = self.calculateSD(height, width, attention, orientation, "15")
            case "XL":
                newheight, newwidth, divisor = self.calculateSD(height, width, attention, orientation, "XL")
            case "FLUX":
                newheight, newwidth, divisor = self.calculateFLUX(height, width, orientation)

        method = ""
        if(divisor > 1): #downscaling
            method = "area"
        else:
            method = "bicubic"
        samples = image.movedim(-1,1)
        s = comfy.utils.common_upscale(samples, newwidth, newheight, method, "disabled")
        s = s.movedim(1,-1)
        return(s,)
    
    def calculateSD(self, height, width, attention, orientation, platform):
        c = 512.0 if platform == "15" else 1024.0 if platform == "XL" else 0.0
        d = (height / c if attention == "minimize" else width / c) if height > width else (width / c if attention == "minimize" else height / c)
        nh, nw = int(height / d), int(width / d)
        # swap if necessary
        if (orientation == "portrait" and nw > nh) or (orientation == "landscape" and nh > nw):
            nh, nw = nw, nh

        return nh, nw, d
    
    def calculateFLUX(s,h,w,o):
        p = h*w
        pr = math.sqrt(p/1048576.0)
        nh = (h/pr)
        nw = (w/pr)
        psv = [896.0, 832.0, 768.0, 640.0]
        plv = [1152.0, 1216.0, 1344.0, 1536.0]
        s = min(nh,nw)
        cs = min(psv, key=lambda x: abs(s-x))
        splv = min(plv)
        nh, nw = (cs, nw*(cs/nh)) if s == nh else (nh*(cs/nw), cs)
        l = max(nh,nw)
        if l > splv:
            cl = min(plv, key=lambda x: abs(l-x))
            nh, nw = (cl, nw*(cl/nh)) if l == nh else (nh*(cl/nw), cl)
        if (o == "portrait" and nw > nh) or (o == "landscape" and nh > nw):
            nh, nw = nw, nh
        pr = math.sqrt(p/(nh*nw))

        return round(nh), round(nw), pr
    
class ImageToPrompt:
    ### Extract image information from PNG Metadata
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        inputs = {
            "required": {"image": (sorted(files), {"image_upload": True})},
            "optional": {}
        }
        return inputs
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "FLOAT", "INT", "INT", "INT",)
    RETURN_NAMES = ("POSITIVE", "NEGATIVE", "HI-RES_PROMPT", "SEED", "STEPS", "CFG", "HEIGHT", "WIDTH", "CLIP_SKIP",)
    FUNCTION = "qmGetMetadata"
    CATEGORY = "QuadmoonNodes/Converters"
    
    def qmGetMetadata(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = Image.open(image_path)

        # get the metadata
        metadata = img.info

        # Check if there are any keys like 'parameters' (this is where the A1111 data is stored)
        parameters = metadata.get("parameters", "No parameters metadata found.")

        if parameters != "No parameters metadata found.":
            # Extract Prompt
            prompt = parameters.split("Negative prompt:")[0].strip()

            # Extract Negative Prompt
            negative_prompt_match = re.search(r"Negative prompt: (.+?)(?=Steps:)", parameters, re.DOTALL)
            negative_prompt = negative_prompt_match.group(1).strip() if negative_prompt_match else None

            # Extract High-res Prompt
            hires_prompt_match = re.search(r"Hires prompt: \"(.+?)\"", parameters, re.DOTALL)
            hires_prompt = hires_prompt_match.group(1).strip() if hires_prompt_match else None

            # Extract other fields
            seed = re.search(r"Seed: (\d+)", parameters).group(1)
            steps = re.search(r"Steps: (\d+)", parameters).group(1)
            cfg_scale = re.search(r"CFG scale: (\d+)", parameters).group(1)
            image_size = re.search(r"Size:\s*(\d+)x(\d+)", parameters)
            if image_size:
                width, height = image_size.groups()
            clip_skip = re.search(r"Clip skip: (\d+)", parameters).group(1)

            # ("POSITIVE", "NEGATIVE", "HI-RES_PROMPT", "SEED", "STEPS", "CFG", "HEIGHT", "WIDTH", "CLIP_SKIP",)
            return(prompt, negative_prompt, hires_prompt, int(seed), int(steps), float(cfg_scale), int(height), int(width), -int(clip_skip),)