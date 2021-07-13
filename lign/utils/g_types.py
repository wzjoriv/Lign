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

if sys.version < "3.9":
    Callable = typing.Callable
    Dict = typing.Dict
    List = typing.List
    Tuple = typing.Tuple
    Set = typing.Set
else:
    Callable = abc.Callable
    Dict = dict
    List = list
    Tuple = tuple
    Set = set

T = TypeVar('T')
