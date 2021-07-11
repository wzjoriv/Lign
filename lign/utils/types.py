from __future__ import annotations
from torch.functional import Tensor
from collections import abc
import sys

import typing
from typing import TypeVar

try:
    from typing import TypedDict
except Exception:
    from typing_extensions import TypedDict

if sys.version > (3, 9):
    Callable = abc.Callable
    Dict = abc.dict
    List = abc.list
    Tuple = abc.tuple
else:
    Callable = typing.Callable
    Dict = typing.Dict
    List = typing.List
    Tuple = typing.Tuple

T = TypeVar('T')
