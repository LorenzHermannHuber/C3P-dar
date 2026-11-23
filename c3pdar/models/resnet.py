"""ResNet-50 C3P-dar model: the DINO model with a shared ImageNet ResNet-50 instead of DINOv3 + LoRA.

The last ResNet stage is dilated (stride 16, as the DINO patches); a shared 1x1 projection maps
2048 -> ``projection_channels`` per modality before concatenation, neck and head.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50

from .dino import RoPETransformerNeck
from .heads import AttentionPixelShuffleHead


class ResNetPPP(nn.Module):
    """(B,3,H,W) x3 condition maps in [-1,1] -> (B,3,H*s,W*s) [log-intensity, raw RCS, raw Doppler].

    ``checkpoint_path`` loads torchvision ImageNet weights; empty means random init.
    """

    def __init__(self, checkpoint_path="", projection_channels=384, neck_layers=12, neck_heads=9,
                 mlp_ratio=4.0, head=None, grad_checkpointing=False):
        super().__init__()
        self.backbone = resnet50(weights=None, replace_stride_with_dilation=[False, False, True])
        if checkpoint_path:
            if not Path(checkpoint_path).is_file():
                raise FileNotFoundError(f"ResNet-50 weights not found: {checkpoint_path}")
            self.backbone.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()
        self.projection = nn.Conv2d(2048, projection_channels, 1, bias=False)
        dim = 3 * projection_channels
        self.neck = RoPETransformerNeck(dim, neck_layers, neck_heads, mlp_ratio=mlp_ratio,
                                        grad_checkpointing=grad_checkpointing)
        self.head = AttentionPixelShuffleHead(dim, grad_checkpointing=grad_checkpointing, **(head or {}))
        self.output_scale = self.head.output_scale

    def forward(self, bev_color_map, bev_seg_map, bev_velocity_map):
        height, width = bev_color_map.shape[-2:]
        batch_size = bev_color_map.shape[0]
        x = F.pad(torch.cat([bev_color_map, bev_seg_map, bev_velocity_map], dim=0), (0, -width % 16, 0, -height % 16))
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = self.projection(net.layer4(net.layer3(net.layer2(net.layer1(x)))))
        x = torch.cat(x.split(batch_size, dim=0), dim=1)
        x = self.head(self.neck(x))
        return x[..., :height * self.output_scale, :width * self.output_scale]
