from __future__ import annotations
from .autodiff import Context
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Type, Sequence

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_functions import Function

@dataclass
class History:
    last_fn: Optional[Type[Function]] = None
    ctx: Optional[Context] = None
    inputs: Sequence[Tensor] = ()