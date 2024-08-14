import os
import sys
import json
import hashlib
import re

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "comfy"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))

import comfy.sd

import folder_paths

SMARTNODESNAME="QuadmoonNodes/SmartNodes"

def check_config_file(filename):
            if os.path.exists(filename):
                return True
            else:
                return False
            
def create_empty_config(filename):
    with open(filename, 'w') as jsonfile:
        json.dump({}, jsonfile)
    print("Quadmoon's Smart Nodes couldn't find the config file! A new one has been created!")

def update_or_create_config(jsonfile, trigger, prompt="", negative="", other_data=""):
    data = {}
    with open(jsonfile) as file:
        data = json.load(file)
    if trigger in data:
        for item in data[trigger]:
            if item.get("other_data") == other_data:
                if prompt != "":
                    item["prompt"] = prompt
                if negative != "":
                    item["negative"] = negative
                break  # Stop iterating once we've found and updated the matching dictionary
        else:  # This else block runs if the loop completes without finding a matching dictionary
            data[trigger].append({"prompt": prompt, "negative": negative, "other_data": other_data})
    else:
        data[trigger] = [{"prompt": prompt, "negative": negative, "other_data": other_data}]
    with open(jsonfile, 'w') as outfile:
        json.dump(data, outfile, indent=4)

def get_list_of_configs(json_file):
    other_data_values = []

    # Check if the JSON file exists
    if not os.path.exists(json_file):
        print("JSON file does not exist.")
        return other_data_values

    # Load the JSON data from the file
    with open(json_file) as file:
        data = json.load(file)

    # Iterate through the JSON data and extract the "other_data" values
    for trigger, config_list in data.items():
        for config_data in config_list:
            other_data_value = config_data.get("other_data")
            if other_data_value:
                other_data_values.append(other_data_value)

    return other_data_values

config_file = "qmSmartNodesConfig.json"

### model loader with trigger for smart nodes
class qmModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                             }}
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "SMART_TRIGGER",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "trigger",)
    FUNCTION = "qmCheckpoint"

    CATEGORY = SMARTNODESNAME

    def qmCheckpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        modelpatcher = out[0]
        clip = out[1]
        vae = out[2]
        # We'll use the hash of the model name as a trigger.
        # Renaming the model will cause any saved data for the model to be lost until the config is edited manually or new data is saved
        trigger = hashlib.sha256(ckpt_name.encode()).hexdigest()+"|"+ckpt_name
        return (modelpatcher, clip, vae, trigger,)
    
class qmSmartNegative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "trigger": ("SMART_TRIGGER",),
            "config_name": ("STRING", {"forceInput": True}),
            "clip": ("CLIP", {"forceInput": True})
        },
        "optional": {
            "optional_text": ("STRING", {"default": "text, watermark, ", "multiline": True,}),
        }}
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("NEGATIVE -",)
    FUNCTION = "qmLoadNegative"

    CATEGORY = SMARTNODESNAME

    def qmLoadNegative(self, trigger, config_name, clip, optional_text=None):
        negative = ""
        triggers = trigger.split("|")
        hash = triggers[0]
        if optional_text is None:
            optional_text = ""

        if check_config_file(config_file):
            # Load the file contents and search for the matching config
            with open(config_file) as configf:
                allConfigs = json.load(configf) #load the JSON
                for hash_value, config_data in allConfigs.items(): #iterate through the json
                    if hash_value == hash: 
                        for config in config_data:
                            if config.get("other_data") == config_name:
                                negative = config.get("negative")
                                break
                        if negative:
                            break #stop when we find the config
            if (negative is None or negative == "") and optional_text != "":
                negative = optional_text
        else:
            create_empty_config(config_file)
        if negative is None or negative == "":
            return ("",)
        tokens = clip.tokenize(negative)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

class qmSmartPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt_text": ("STRING", {"default": "", "multiline": True,}),
            "config_name": ("STRING", {"forceInput": True}),
            "trigger": ("SMART_TRIGGER",),
            "clip": ("CLIP", {"forceInput": True}),
        }}
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("POSITIVE +",)
    FUNCTION = "qmLoadPrompt"

    CATEGORY = SMARTNODESNAME

    def qmLoadPrompt(self, prompt_text, config_name, trigger, clip):
        qmprompt = ""
        noswap = False

        def load_value(json_data, trigger_value, key):
            if trigger_value in json_data:
                for config in json_data[trigger_value]:
                    if config.get("other_data") == config_name:
                        return config.get(key)
            return ""

        if check_config_file(config_file):
            # Load the file contents and search for the matching config
            with open(config_file) as configf:
                allConfigs = json.load(configf)
                triggers = trigger.split("|")
                hash = triggers[0]
                prompt_template = load_value(allConfigs, hash, "prompt")
                if prompt_template == "":
                    qmprompt = prompt_text
                    noswap = True
                pattern = re.compile(r'\b{}\b'.format(re.escape(hash)))
                if not noswap:
                    qmprompt = re.sub(pattern, prompt_text, prompt_template)
        else:
            create_empty_config(config_file)
        tokens = clip.tokenize(qmprompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )
    
class qmSaveNegative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "negative": ("STRING", {"forceInput": True}),
            "trigger": ("SMART_TRIGGER",),
            "config_name": ("STRING", {"forceInput": True})
        }}
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    FUNCTION = "qmSaveNegative"

    CATEGORY = SMARTNODESNAME

    def qmSaveNegative(self, negative, trigger, config_name):
        if not check_config_file(config_file):
            create_empty_config(config_file)
        triggers = trigger.split("|")
        hash = triggers[0]
        ckpt_name = triggers[1]
        otherdata = ckpt_name + " - " + config_name
        update_or_create_config(jsonfile=config_file, negative=negative, trigger=hash, other_data=otherdata)
        return (negative,)

class qmSavePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "prompt_start": ("STRING", {"forceInput": True}),
            "image_content": ("STRING", {"default": "", "multiline": True,}),
            "prompt_end": ("STRING", {"forceInput": True}),
            "trigger": ("SMART_TRIGGER",),
            "config_name": ("STRING", {"forceInput": True})
        }}
    
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("FULL_PROMPT","PROMPT_TEXT")
    FUNCTION = "qmSavePrompt"

    CATEGORY = SMARTNODESNAME

    def qmSavePrompt(self, prompt_start, image_content, prompt_end, trigger, config_name):
        output = prompt_start + ", " + image_content + ", " + prompt_end
        prompt = prompt_start + ", " + trigger + ", " + prompt_end
        if not check_config_file(config_file):
            create_empty_config(config_file)
        triggers = trigger.split("|")
        hash = triggers[0]
        ckpt_name = triggers[1]
        otherdata = ckpt_name + " - " + config_name
        update_or_create_config(jsonfile=config_file, prompt=prompt, trigger=hash, other_data=otherdata)
        return (output,image_content,)

class qmLoadConfigs:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "config_names": (get_list_of_configs(config_file), ),
            }
        }
    
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("CONFIG_NAME", )
    FUNCTION = "qmLoadConfig"

    CATEGORY = SMARTNODESNAME

    def qmLoadConfig(self, config_names):
        return (config_names,)