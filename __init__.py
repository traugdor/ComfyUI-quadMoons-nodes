"""
@author: quadmoon (https://github.com/traugdor)
@title: quadmoon's ComfyUI nodes
@nickname: quadmoon's Nodes
@description: These are just some nodes I wanted and couldn't find where anyone else had made them yet.
"""

from .theButton import quadmoonThebutton
from .clipTextEncoder import quadmoonCLIPTextEncode, quadmoonCLIPTextEncode2
from .converters import IntToString, FloatToString, BoolToString, NumberToString, NormalizeHW, ImageToPrompt
from .conditionals import INTCompare
from .samplers import qmKSampler, qmKSamplerAdvanced, qmRotationalSampler, qmLatentImage, qmKSamplerBatched
from .smartnodes import qmModelLoader, qmLoadConfigs, qmSmartPrompt, qmSmartNegative, qmSavePrompt, qmSaveNegative
from .changeBackground import qmChangeBackground
from .createBatchFromLatent import qmBatchFromLatent

import os
import sys
import configparser
import shutil

def add_submodules_to_path_and_remove_init():
    """Read .gitmodules, extract submodule paths, add to sys.path, and remove unnecessary files."""
    repo_root = os.path.dirname(__file__)  # Adjust as needed to locate your repo root
    gitmodules_path = os.path.join(repo_root, ".gitmodules")

    if not os.path.exists(gitmodules_path):
        print(".gitmodules file not found. Skipping submodule imports.")
        return

    # Essential files/directories to keep
    essential_items = {
        'efficiency_nodes.py',
        'node_settings.json',
        'tsc_utils.py',
        '__init__.py',
        'py',  # directory containing necessary Python files
        'efficiency_nodes',
        'workflows',
        '__pycache__'
    }

    # Use ConfigParser to parse .gitmodules
    config = configparser.ConfigParser()
    config.read(gitmodules_path)

    for section in config.sections():
        if section.startswith("submodule"):
            submodule_path = config[section].get("path")
            if submodule_path:
                full_path = os.path.join(repo_root, submodule_path)
                
                # Check if this is the efficiency-nodes directory
                if submodule_path == "efficiency-nodes":
                    # Get the new path with underscore
                    new_path = os.path.join(repo_root, "efficiency_nodes")
                    
                    # Only rename if the underscore version doesn't exist
                    if not os.path.exists(new_path):
                        try:
                            os.rename(full_path, new_path)
                            print(f"Renamed {full_path} to {new_path}")
                            full_path = new_path  # Update the path for further processing
                        except Exception as e:
                            print(f"Warning: Could not rename efficiency-nodes directory: {e}")
                    else:
                        # If underscore version exists, use that path instead
                        full_path = new_path
                
                if os.path.exists(full_path):
                    # Clean up non-essential files
                    for item in os.listdir(full_path):
                        if item not in essential_items:
                            item_path = os.path.join(full_path, item)
                            try:
                                if os.path.isfile(item_path):
                                    os.remove(item_path)
                                    print(f"Removed unnecessary file: {item}")
                                elif os.path.isdir(item_path):
                                    shutil.rmtree(item_path)
                                    print(f"Removed unnecessary directory: {item}")
                            except Exception as e:
                                print(f"Warning: Could not remove {item}: {e}")
                    init_file = os.path.join(full_path, '__init__.py')
                    with open(init_file, 'w') as f:
                        # Write a special comment that ComfyUI will ignore
                        f.write("# This is an empty __init__.py file to make this directory a package.\n")
                        f.write("# It intentionally contains no NODE_CLASS_MAPPINGS to prevent ComfyUI from loading it.\n")


                # Add the submodule path to sys.path
                if full_path not in sys.path:
                    sys.path.append(full_path)
                    print(f"Added submodule to path: {full_path}")

# Call the function to add submodules to sys.path and handle __init__.py
add_submodules_to_path_and_remove_init()

def get_encode_prompts():
    from .efficiency_nodes.efficiency_nodes import encode_prompts
    return encode_prompts

NODE_CLASS_MAPPINGS = {
    "quadmoonThebutton": quadmoonThebutton,
    "quadmoonCLIPTextEncode": quadmoonCLIPTextEncode,
    "quadmoonCLIPTextEncode2": quadmoonCLIPTextEncode2,
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
    "quadmoonCLIPTextEncode2": "CLIP Text Encode - Advanced",
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
