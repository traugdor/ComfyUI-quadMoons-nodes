import os
import sys

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
                "clip": ("CLIP", {"forceInput": True}),
                "text": ("STRING", {"forceInput": True})
            }
        }

        return inputs

    CATEGORY = "QuadmoonNodes"
    RETURN_TYPES=("CONDITIONING",)
    FUNCTION = "qmTextEncode"

    def qmTextEncode(self, clip, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )