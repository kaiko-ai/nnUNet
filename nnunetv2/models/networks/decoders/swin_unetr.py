"""Decoder based on Swin UNETR."""

from typing import List

import torch
from monai.networks.blocks import dynunet_block, unetr_block
from torch import nn

from nnunetv2.model_sharing.hub import load_state_dict_from_url


class SwinUNETRDecoder(nn.Module):
    """Swin transformer decoder based on UNETR [0].

    - [0] UNETR: Transformers for 3D Medical Image Segmentation
      https://arxiv.org/pdf/2103.10504
    """

    def __init__(
        self,
        out_channels: int,
        feature_size: int = 48,
        spatial_dims: int = 3,
        checkpoint_path: str | None = None,
    ) -> None:
        """Builds the decoder.

        Args:
            out_channels: Number of output channels.
            feature_size: Dimension of network feature size.
            spatial_dims: Number of spatial dimensions.
            checkpoint_path: Path to the checkpoint file.
        """
        super().__init__()

        self._checkpoint_path = checkpoint_path

        self.decoder5 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder4 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder3 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder2 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.decoder1 = unetr_block.UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name="instance",
            res_block=True,
        )
        self.out = dynunet_block.UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

        self._load_checkpoint()

    def _forward_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward function for multi-level feature maps to a single one."""
        enc0, enc1, enc2, enc3, hid3, dec4 = features
        dec3 = self.decoder5(dec4, hid3)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        return self.out(out)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Maps the patch embeddings to a segmentation mask.

        Args:
            features: List of multi-level intermediate features from
                :class:`SwinUNETREncoder`.

        Returns:
            Tensor containing scores for all of the classes with shape
            (batch_size, n_classes, image_height, image_width).
        """
        return self._forward_features(features)

    def _load_checkpoint(self) -> None:
        if self._checkpoint_path is not None:
            state_dict = load_state_dict_from_url(self._checkpoint_path, force=True)
            self.load_state_dict(state_dict)
