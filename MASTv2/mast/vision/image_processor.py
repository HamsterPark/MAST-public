"""STM image preprocessor for the VIGIL DINOv3 pipeline.

Pure interface — bypasses HuggingFace ``AutoImageProcessor`` (which would
quantise to uint8 and lose the 16-bit pm dynamic range) and yields a tensor
already in DINOv3's expected mean/std space.

Strategy (mirrors VIGIL §3 of the implementation guide):
    1. Robust per-image normalisation: median + MAD (1.4826).
    2. Clip to ±5 σ to suppress scan artefacts.
    3. tanh-squash to [0, 1].
    4. Replicate to 3 channels.
    5. Resize to ``target_size`` (defaults to 512, must be multiple of 16).
    6. Apply ImageNet mean/std (DINOv3 inherits DINOv2 stats).

The processor is torch-only at call time; importing the module does NOT
import torch (so callers with non-vision dependencies on ``mast.vision``
keep working).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor


_DINO_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_DINO_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


class STMImageProcessor:
    """Float32 pm tensor → DINOv3 input tensor.

    Args:
        target_size: square target side length in pixels. Default 512.
            DINOv3-L/16 patches are 16 px so target_size must be a multiple
            of 16. The default 512 yields a 32×32 patch grid.
        clip_sigma:  clip per-image z-scores to ±clip_sigma. Default 5.0.
        squash:      tanh squash compression. 0 disables (linear). Default 0.3.
    """

    DINO_MEAN: tuple[float, float, float] = _DINO_MEAN
    DINO_STD: tuple[float, float, float] = _DINO_STD

    def __init__(
        self,
        target_size: int = 512,
        clip_sigma: float = 5.0,
        squash: float = 0.3,
        patch_size: int = 16,
    ) -> None:
        if target_size % patch_size != 0:
            raise ValueError(
                f"target_size ({target_size}) must be a multiple of patch_size "
                f"({patch_size}); DINOv3 patch grid would be non-integer otherwise."
            )
        self.target_size = target_size
        self.clip_sigma = clip_sigma
        self.squash = squash
        self.patch_size = patch_size
        self._mean: Any = None  # lazy torch tensor
        self._std: Any = None

    def _ensure_stats(self, device: Any) -> None:
        import torch

        if self._mean is None:
            self._mean = torch.tensor(self.DINO_MEAN).view(1, 3, 1, 1)
            self._std = torch.tensor(self.DINO_STD).view(1, 3, 1, 1)
        if self._mean.device != device:
            self._mean = self._mean.to(device)
            self._std = self._std.to(device)

    def __call__(self, image_pm: "Tensor") -> "Tensor":
        """Run the pipeline.

        Args:
            image_pm: float tensor of shape (H, W) or (B, H, W). Units = pm.

        Returns:
            (B, 3, target_size, target_size) float tensor in DINOv3 input space.
        """
        import torch
        import torch.nn.functional as F

        if image_pm.dim() == 2:
            x = image_pm.unsqueeze(0)
        elif image_pm.dim() == 3:
            x = image_pm
        else:
            raise ValueError(
                f"image_pm must be (H,W) or (B,H,W); got {tuple(image_pm.shape)}"
            )

        x = x.float()

        # Robust per-image median / MAD
        flat = x.reshape(x.shape[0], -1)
        med = flat.median(dim=1).values.view(-1, 1, 1)
        mad = (
            (flat - med.view(-1, 1)).abs().median(dim=1).values.view(-1, 1, 1)
            * 1.4826
            + 1e-6
        )
        z = (x - med) / mad

        # Clip extreme outliers
        z = z.clamp(-float(self.clip_sigma), float(self.clip_sigma))

        # tanh squash to [0, 1]; squash=0 means linear → just shift to [0, 1]
        if self.squash > 0:
            z = torch.tanh(z * float(self.squash)) * 0.5 + 0.5
        else:
            z = (z / (2 * float(self.clip_sigma))) + 0.5

        # 1 channel → 3 channels
        x3 = z.unsqueeze(1).expand(-1, 3, -1, -1)

        # Resize if needed
        if x3.shape[-1] != self.target_size or x3.shape[-2] != self.target_size:
            x3 = F.interpolate(
                x3,
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            )

        # ImageNet normalisation
        self._ensure_stats(x3.device)
        x3 = (x3 - self._mean) / self._std
        return x3


__all__ = ["STMImageProcessor"]
