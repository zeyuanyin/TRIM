from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentSelector(nn.Module):
    """Pairwise latent selector for TRIM's trajectory reduction.

    Given two intermediate multi-view Gaussian latents z1, z2 (B, V, C, H, W), predicts
    which one leads to the higher-quality final render. `num_conv_layers=1,
    num_mlp_layers=2, use_prompt=False` ("Conv1-FC2") is the default used for TRIM's main
    results; other combinations reproduce the Table 3 ablation.
    """

    def __init__(
        self,
        num_conv_layers: int = 1,
        num_mlp_layers: int = 2,
        use_prompt: bool = False,
        in_channels: int = 16,
        num_views: int = 4,
        prompt_dim: int = 512,
    ):
        super().__init__()
        assert num_conv_layers in (0, 1, 2, 3)
        assert num_mlp_layers in (2, 3)
        self.use_prompt = use_prompt
        self.view_to_one = nn.Linear(num_views, 1)

        channels = [in_channels, 32, 64, 128][: num_conv_layers + 1]
        conv = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            conv += [nn.Conv2d(c_in, c_out, 3, stride=2, padding=1), nn.GELU(), nn.Dropout2d(0.2)]
        self.encoder = nn.Sequential(*conv) if conv else nn.Identity()
        feat_dim = channels[-1]

        if use_prompt:
            head_in = feat_dim + prompt_dim
            head = []
        else:
            head_in = feat_dim
            head = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        for i in range(num_mlp_layers):
            out_dim = 1 if i == num_mlp_layers - 1 else head_in
            head.append(nn.Linear(head_in, out_dim))
            if i < num_mlp_layers - 1:
                head.append(nn.GELU())
            head_in = out_dim
        self.classifier = nn.Sequential(*head)

    def _encode(self, z: torch.Tensor) -> torch.Tensor:
        bs, v, c, h, w = z.shape
        z = z.permute(0, 2, 3, 4, 1).reshape(-1, v)
        z = self.view_to_one(z).reshape(bs, c, h, w)
        return self.encoder(z)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor, prompt: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = self._encode(z1) - self._encode(z2)
        if self.use_prompt:
            diff = F.adaptive_avg_pool2d(diff, 1).flatten(1)
            diff = torch.cat([diff, prompt], dim=1)
        return self.classifier(diff).squeeze(1)
