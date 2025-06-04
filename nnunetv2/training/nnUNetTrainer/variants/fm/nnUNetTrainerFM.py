from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing_extensions import override
from typing import Union, List, Tuple
import torch
from torch import nn
from nnunetv2.models.networks.encoders.voco import VoCoB
from nnunetv2.models.networks.decoders.swin_unetr import SwinUNETRDecoder
from nnunetv2.models.networks.composed.encoder_decoder import EncoderDecoder


class nnUNetTrainerFM(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device=None):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.configuration_manager.configuration['architecture']['network_class_name'] = "voco_b"
        self.configuration_manager.configuration['patch_size'] = [96, 96, 96]
        self.enable_deep_supervision = False
        
        self.freeze_encoder = True  # Freeze the encoder by default
        self.num_epochs = 2000

    @override
    def build_network_architecture(self, architecture_class_name: str,
                                arch_init_kwargs: dict,
                                arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                num_input_channels: int,
                                num_output_channels: int,
                                enable_deep_supervision: bool = True) -> nn.Module:
        
        match architecture_class_name:
            case "voco_b":
                return EncoderDecoder(
                    encoder=VoCoB(out_indices=6),
                    decoder=SwinUNETRDecoder(
                        out_channels=num_output_channels,
                        feature_size=48,
                        spatial_dims=3,
                    ),
                    freeze_encoder=self.freeze_encoder,
                )
            case "coralbay":
                raise NotImplementedError("Architecture is not implemented yet.")
            case _:
                raise ValueError(f"Unknown architecture class name: {architecture_class_name}")

    @override
    def set_deep_supervision_enabled(self, enabled: bool):
        pass

    @override
    def _do_i_compile(self) -> bool:
        # Disable torch.compile for custom architecture to avoid symbolic tracing issues
        return False