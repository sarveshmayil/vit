import torch
import torch.nn as nn

from vit.utils import patchify

class ViT(nn.Module):
    def __init__(self, n_patches: int):
        super(ViT, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        patches = patchify(input, self.n_patches)
        return patches