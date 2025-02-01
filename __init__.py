"""
@author: quadmoon (https://github.com/traugdor)
@title: quadmoon's ComfyUI nodes
@nickname: quadmoon's Nodes
@description: These are just some nodes I wanted and couldn't find where anyone else had made them yet.
"""

from .theButton import quadmoonThebutton
from .clipTextEncoder import quadmoonCLIPTextEncode
from .converters import IntToString, FloatToString, BoolToString, NumberToString, NormalizeHW, ImageToPrompt
from .conditionals import INTCompare
from .samplers import qmKSampler, qmKSamplerAdvanced, qmRotationalSampler, qmLatentImage, qmKSamplerBatched
from .smartnodes import qmModelLoader, qmLoadConfigs, qmSmartPrompt, qmSmartNegative, qmSavePrompt, qmSaveNegative
from .changeBackground import qmChangeBackground
from .createBatchFromLatent import qmBatchFromLatent

NODE_CLASS_MAPPINGS = {
    "quadmoonThebutton": quadmoonThebutton,
    "quadmoonCLIPTextEncode": quadmoonCLIPTextEncode,
    "quadmoonConvertIntToString": IntToString,
    "quadmoonConvertFloatToString": FloatToString,
    "quadmoonConvertBoolToString": BoolToString,
    "quadmoonConvertNumberToString": NumberToString,
    "quadmoonConvertImageToPrompt": ImageToPrompt,
    "quadmoonINTConditionalOperation": INTCompare,
    "quadmoonConvertNormalizeHW": NormalizeHW,
    "quadmoonKSampler": qmKSampler,
    "quadmoonKSamplerAdvanced": qmKSamplerAdvanced,
    "quadmoonRotationalSampler": qmRotationalSampler,
    "quadmoonModelLoader": qmModelLoader,
    "quadmoonLoadConfigs": qmLoadConfigs,
    "quadmoonSmartPrompt": qmSmartPrompt,
    "quadmoonSmartNeg": qmSmartNegative,
    "quadmoonSavePrompt": qmSavePrompt,
    "quadmoonSaveNeg": qmSaveNegative,
    "quadmoonChangeBackground": qmChangeBackground,
    "quadmoonLatentImage": qmLatentImage,
    "quadmoonBatchFromLatent": qmBatchFromLatent,
    "quadmoonKSamplerBatched": qmKSamplerBatched,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "quadmoonThebutton": "The BUTTON",
    "quadmoonCLIPTextEncode": "CLIP exText Encode",
    "quadmoonConvertIntToString": "Convert Integer to String",
    "quadmoonConvertFloatToString": "Convert Float to String",
    "quadmoonConvertBoolToString": "Convert Bool to String",
    "quadmoonConvertNumberToString": "Convert Number to String",
    "quadmoonConvertImageToPrompt": "Get A1111 Prompt from Image",
    "quadmoonINTConditionalOperation": "INT Conditional Operation",
    "quadmoonConvertNormalizeHW": "Normalize Image Dimensions",
    "quadmoonKSampler": "KSampler - Extra Outputs",
    "quadmoonKSamplerAdvanced": "KSamplerAdvanced - Extra Outputs",
    "quadmoonRotationalSampler": "KSamplerAdvanced - Alternating Sampling",
    "quadmoonModelLoader": "Load Model - Smart",
    "quadmoonLoadConfigs": "Load Config - Smart",
    "quadmoonSmartPrompt": "CLIPTextEncode - Smart Prompt",
    "quadmoonSmartNeg": "CLIPTextEncode - Smart Negative",
    "quadmoonSavePrompt": "Save Smart Prompt Config",
    "quadmoonSaveNeg": "Save Smart Negative Config",
    "quadmoonChangeBackground": "Change Background of Image (SEGM SEGS)",
    "quadmoonLatentImage": "Empty Latent Image (orientation)",
    "quadmoonBatchFromLatent": "Batch From Latent Image",
    "quadmoonKSamplerBatched": "KSampler For Each",
}

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
