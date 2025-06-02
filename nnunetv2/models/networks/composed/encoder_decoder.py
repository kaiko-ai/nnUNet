"""Encoder-Decoder architecture."""

import torch
from monai.inferers.inferer import Inferer
from torch import nn


class EncoderDecoder(nn.Module):
    """EncoderDecoder composed network.

    A neural network architecture combining an encoder and a decoder.
    Optionally integrates MONAI's `Inferer` for optimized inference.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        inferer: Inferer | None = None,
    ) -> None:
        """Build the encoder-decoder network.

        Example:
        ```
        >>> model = kaiko.radiology_fm.models.networks.EncoderDecoder(
        >>>     encoder=kaiko.radiology_fm.models.networks.SwinUNETREncoder(
        >>>         out_indices=6,
        >>>     ),
        >>>     decoder=kaiko.radiology_fm.models.networks.SwinUNETRDecoder(
        >>>         out_channels=3,
        >>>     ),
        >>>     inferer=monai.inferers.SlidingWindowInferer(
        >>>         roi_size=(64, 64, 64),
        >>>         sw_batch_size=2,
        >>>         overlap=0.75,
        >>>     ),
        >>> )
        ```

        Args:
            encoder: The encoder module, responsible for
                extracting features from the input tensor.
            decoder: The decoder module, responsible for
                reconstructing or transforming the features
                into the desired output.
            inferer: An optional MONAI `Inferer` for efficient
                inference during evaluation.
        """
        super().__init__()

        self._encoder = encoder
        self._decoder = decoder
        self._inferer = inferer

    def forward_networks(self, tensor: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the encoder and decoder.

        Args:
            tensor:  Input tensor to be processed.

        Returns:
            Output tensor after encoding and decoding.
        """
        patch_embeddings = self._encoder(tensor)
        return self._decoder(patch_embeddings)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the network.

        During training, applies the encoder and decoder directly.
        During inference, uses the provided inferer for optimized processing if available.

        Args:
            tensor: Input tensor to be processed.

        Returns: Output tensor after processing.
        """
        if not self.training and self._inferer:
            return self._inferer(inputs=tensor, network=self.forward_networks)

        return self.forward_networks(tensor)
