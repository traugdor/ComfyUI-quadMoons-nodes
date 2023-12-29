import json
from pathlib import Path
from server import PromptServer
import server

class quadmoonThebutton:
    # Tired of moving your mouse *ALL THE WAY OVER* to the Cancel Button for your queue?
    # This button provides an easier place for you to kill your workflow if you see something has gone wrong.
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {}

        return inputs

    CATEGORY = "QuadmoonNodes"
    RETURN_TYPES=()