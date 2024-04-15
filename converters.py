from typing import TypeAlias

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
                "height": ("INT", {}),
                "width": ("INT", {}),
                "platform":(["SD1.5", "SDXL"],),
                "orientation":(["portrait", "landscape"],)
            }
        }
        return inputs
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("HEIGHT", "WIDTH",)
    FUNCTION = "qmNormalizeHW"
    CATEGORY = "QuadmoonNodes/Converters"

    def qmNormalizeHW(self, height, width, platform, orientation):
        divisor = 0.0
        newheight = 0
        newwidth = 0
        temp = 0
        if(platform == "SD1.5"):
            if(height > width):
                divisor = height/512.0
            else:
                divisor = width/512.0
        else:
            if(height > width):
                divisor = height/1024.0
            else:
                divisor = width/1024.0
        newheight = height/divisor
        newwidth = width/divisor
        if((orientation == "portrait" and newheight < newwidth) or (orientation == "landscape" and newwidth < newheight)):
            temp = newheight
            newheight = newwidth
            newwidth = temp
        return(int(newheight), int(newwidth),)