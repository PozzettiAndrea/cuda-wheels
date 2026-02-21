# pyre-ignore-all-errors
import torch


def generic_nms(
    dets: torch.Tensor, scores: torch.Tensor, iou_threshold: float, use_iou_matrix: bool
) -> torch.Tensor:
    from . import _C  # pyre-ignore[21]

    return _C.generic_nms(
        dets.contiguous(), scores.contiguous(), iou_threshold, use_iou_matrix
    )  # pyre-ignore[16]
