import torch
import torch.nn as nn
from diffusers import UNet2DModel

class ConditionalDDPM(nn.Module):
    def __init__(self, num_classes=24, dim=512):
        super().__init__()
        channel = dim // 4
        self.ddpm = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=[channel, channel, channel*2, channel*2, channel*4],
            down_block_types=["DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"],
            up_block_types=["UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"],
            class_embed_type="identity",
        )
        self.class_embedding = nn.Sequential(
            nn.Linear(num_classes, dim),
            nn.SiLU()
        )

    def forward(self, x, t, label):
        class_embed = self.class_embedding(label)
        return self.ddpm(sample=x, timestep=t, class_labels=class_embed).sample