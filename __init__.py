"""
@author: quadmoon (https://github.com/traugdor)
@title: quadmoon's ComfyUI nodes
@nickname: quadmoon's Nodes
@description: These are just some nodes I wanted and couldn't find where anyone else had made them yet.
"""

from .theButton import quadmoonThebutton

NODE_CLASS_MAPPINGS = {
    "quadmoonThebutton": quadmoonThebutton,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "quadmoonThebutton": "quadmoon's Nodes - The BUTTON",
}

WEB_DIRECTORY = "js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
