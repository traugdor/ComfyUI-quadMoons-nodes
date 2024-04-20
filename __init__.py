"""
@author: quadmoon (https://github.com/traugdor)
@title: quadmoon's ComfyUI nodes
@nickname: quadmoon's Nodes
@description: These are just some nodes I wanted and couldn't find where anyone else had made them yet.
"""

from .theButton import quadmoonThebutton
from .clipTextEncoder import quadmoonCLIPTextEncode
from .converters import IntToString, FloatToString, BoolToString, NumberToString, NormalizeHW
from .conditionals import INTCompare
from .samplers import qmKSampler, qmKSamplerAdvanced


NODE_CLASS_MAPPINGS = {
    "quadmoonThebutton": quadmoonThebutton,
    "quadmoonCLIPTextEncode": quadmoonCLIPTextEncode,
    "quadmoonConvertIntToString": IntToString,
    "quadmoonConvertFloatToString": FloatToString,
    "quadmoonConvertBoolToString": BoolToString,
    "quadmoonConvertNumberToString": NumberToString,
    "quadmoonINTConditionalOperation": INTCompare,
    "quadmoonConvertNormalizeHW": NormalizeHW,
    "quadmoonKSampler": qmKSampler,
    "quadmoonKSamplerAdvanced": qmKSamplerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "quadmoonThebutton": "The BUTTON",
    "quadmoonCLIPTextEncode": "CLIP exText Encode",
    "quadmoonConvertIntToString": "Convert Integer to String",
    "quadmoonConvertFloatToString": "Convert Float to String",
    "quadmoonConvertBoolToString": "Convert Bool to String",
    "quadmoonConvertNumberToString": "Convert Number to String",
    "quadmoonINTConditionalOperation": "INT Conditional Operation",
    "quadmoonConvertNormalizeHW": "Normalize Image Dimensions",
    "quadmoonKSampler": "KSampler - Extra Outputs",
    "quadmoonKSamplerAdvanced": "KSamplerAdvanced - Extra Outputs",
}

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
