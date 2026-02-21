# pyre-ignore-all-errors
from typing import Tuple

import torch


def get_connected_components(
    x: torch.Tensor, get_counts: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    from . import _C  # pyre-ignore[21]

    return _C.cc_2d(x.contiguous(), get_counts)  # pyre-ignore[16]
