import os
import sys
import importlib.util

# Assuming your script is two folders deep
current_file_directory = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_file_directory, "..", ".."))
comfy_folder_path = os.path.join(project_root, "comfy")

sys.path.insert(0, comfy_folder_path)

class quadmoonCLIPTextEncode:
    # Need to use a node that outputs a STRING but you need some sort of conditioning instead?
    # This node provides an easy way to do just that.
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "tooltip": "A CLIP model used for encoding the text."}),
                "text": ("STRING", {"forceInput": True, "tooltip": "The text to be encoded."})
            }
        }

        return inputs

    CATEGORY = "QuadmoonNodes"
    RETURN_TYPES=("CONDITIONING",)
    OUTPUT_TOOLTIPS=("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "qmTextEncode"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens(tokens, return_pooled=True)

    def qmTextEncode(self, clip, text):
        cond, pooled = self.encode(clip, text)
        return ([[cond, {"pooled_output": pooled}]], )
    
class quadmoonCLIPTextEncode2(quadmoonCLIPTextEncode):
    # check for Efficiency Nodes installed, import that code and use it here

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "clip": ("CLIP", {"forceInput": True, "tooltip": "A CLIP model used for encoding the text."}),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "POSITIVE_PROMPT": ("STRING", {"multiline": True, "tooltip": "The text to be encoded."}),
                "NEGATIVE_PROMPT": ("STRING", {"multiline": True, "tooltip": "The text to be encoded."}),
            },
            "hidden": {
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
            }
        }

        if "weight_interpretation" in inputs["hidden"]:
            inputs["required"]["weight_interpretation"] = inputs["hidden"].pop("weight_interpretation")

        return inputs
    
    CATEGORY = "QuadmoonNodes"
    RETURN_TYPES=("CONDITIONING","CONDITIONING",)
    RETURN_NAMES=("POSITIVE", "NEGATIVE",)
    OUTPUT_TOOLTIPS=("A conditioning containing the embedded text used to guide the diffusion models.","A conditioning containing the embedded text used to guide the diffusion models.")
    FUNCTION = "qmTextEncode2"
    
    def qmTextEncode2(self, clip, clip_skip, POSITIVE_PROMPT, NEGATIVE_PROMPT, weight_interpretation=None):
        from . import get_encode_prompts
        encode_prompts = get_encode_prompts()
        pos, neg, _ = encode_prompts(POSITIVE_PROMPT, NEGATIVE_PROMPT, "none", weight_interpretation, clip, clip_skip,
                                         None,None,None,False,None,None,
                                         "base")
        return (pos, neg,)