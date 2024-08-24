import torch
import torch.nn as nn

from vit.utils import patchify, pair, positional_embedding_2d

from typing import Union, Tuple

class ViT(nn.Module):
    def __init__(self, image_size: Tuple[int, int], hidden_dim: int, channels: int = 3, patch_size: Union[int, Tuple[int, int]] = 16):
        super(ViT, self).__init__()

        self.image_size = pair(image_size)
        im_h, im_w = self.image_size
        self.patch_size = pair(patch_size)
        patch_h, patch_w = self.patch_size

        patch_dim = channels * patch_h * patch_w
        self.linear_projection = nn.Linear(in_features=patch_dim, out_features=hidden_dim)

        self.class_token = nn.Parameter(torch.rand(1, hidden_dim))

        self.pos_embedding = positional_embedding_2d(
            h=im_h // patch_h + im_h % patch_h,
            w=im_w // patch_w + im_w % patch_w,
            dim=hidden_dim
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        device = inp.device

        # (B, C, H, W) -> (B, n_patches, patch_dim)
        x = patchify(inp, self.patch_size)

        # (B, n_patches, patch_dim) -> (B, n_patches, hidden_dim)
        x = self.linear_projection(x)

        # Add class token to the input
        # (B, n_patches, hidden_dim) -> (B, n_patches+1, hidden_dim)
        cls_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x += self.pos_embedding.to(device, dtype=x.dtype)

        return x