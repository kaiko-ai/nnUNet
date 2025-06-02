"""VoCo Self-Supervised Encoders."""

from nnunetv2.models.networks.encoders import swin_unetr
from nnunetv2.model_sharing.hub import load_state_dict_from_url


class _VoCo(swin_unetr.SwinUNETREncoder):
    """Base class for the VoCo self-supervised encoders."""

    _checkpoint: str
    """Path to the model state dict."""

    _md5: str | None = None
    """State dict MD5 validation code."""

    def __init__(self, feature_size: int, out_indices: int | None = None) -> None:
        super().__init__(
            in_channels=1,
            feature_size=feature_size,
            spatial_dims=3,
            out_indices=out_indices,
        )

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads the model checkpoint."""
        # state_dict = hub.load_state_dict_from_url(self._checkpoint)
        state_dict = load_state_dict_from_url(self._checkpoint, md5=self._md5)
        self.load_state_dict(state_dict)


class VoCoB(_VoCo):
    """VoCo Self-supervised pre-trained B model."""

    _checkpoint = "https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt"
    _md5 = "f80c4da2f81d700bdae3df188f2057eb"

    def __init__(self, out_indices: int | None = None) -> None:
        super().__init__(feature_size=48, out_indices=out_indices)
