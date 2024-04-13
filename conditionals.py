import math

from typing import Any, Callable, Mapping

INT_BINARY_CONDITIONS: Mapping[str, Callable[[int, int], bool]] = {
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
    "Gt": lambda a, b: a > b,
    "Lt": lambda a, b: a < b,
    "Geq": lambda a, b: a >= b,
    "Leq": lambda a, b: a <= b,
}

class INTCompare:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "int_a": ("INT", {}),
                "int_b": ("INT", {}),
                "op": (list(INT_BINARY_CONDITIONS.keys()),),
                "if_true_return": (["a","b"],),
                "if_false_return": (["a","b"],)
            }
        }
        return inputs
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("RETURN VALUE",)
    FUNCTION = "qmINTConditional"
    CATEGORY = "QuadmoonNodes/qmConditionals"

    def qmINTConditional(self, int_a: int, int_b: int, op:str, if_true: str, if_false:str) -> tuple[int,]:
        returnval = 0
        if (INT_BINARY_CONDITIONS[op](int_a,int_b)):
            if(if_true == 'a'):
                returnval = int_a
            else:
                returnval = int_b
        else:
            if(if_true == 'a'):
                returnval = int_b
            else:
                returnval = int_a
        return (returnval,)