import torch
import torch.nn as nn

from vit.transformer import Transformer
from vit.utils import patchify, pair, positional_embedding_2d

from typing import Union, Tuple

class ViT(nn.Module):
    def __init__(
        self,
        n_classes: int,
        image_size: Tuple[int, int],
        hidden_dim: int,
        depth: int,
        n_heads: int = 8,
        mlp_dim: int = 128,
        channels: int = 3,
        patch_size: Union[int, Tuple[int, int]] = 8
    ):
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
        ).requires_grad_(False)

        self.transformer = Transformer(
            dim=hidden_dim,
            depth=depth,
            heads=n_heads,
            mlp_dim=mlp_dim
        )

        self.prediction_head = nn.Linear(hidden_dim, n_classes)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, n_patches, patch_dim)
        x = patchify(inp, self.patch_size)

        # (B, n_patches, patch_dim) -> (B, n_patches, hidden_dim)
        x = self.linear_projection(x)

        # Add class token to the input
        # (B, n_patches, hidden_dim) -> (B, n_patches+1, hidden_dim)
        cls_token = self.class_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add positional embedding
        x[:, 1:, :] += self.pos_embedding.to(x.device)

        # Pass through transformer
        # (B, n_patches+1, hidden_dim) -> (B, n_patches+1, hidden_dim)
        x = self.transformer(x)

        # Average pool over the patches
        # (B, n_patches+1, hidden_dim) -> (B, hidden_dim)
        x = x.mean(dim=1)

        # Pass through prediction head
        # (B, hidden_dim) -> (B, n_classes)
        x = self.prediction_head(x)

        return x