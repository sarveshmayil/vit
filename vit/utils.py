import torch
import torch.nn.functional as F

from typing import Union, Tuple

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def patchify(images: torch.Tensor, patch_size: Union[int, Tuple[int, int]]) -> torch.Tensor:
    B, C, H, W = images.shape
    patch_h, patch_w = pair(patch_size)

    # If image dimensions are not divisible by patch size, pad the image
    pad_h = (patch_h - H % patch_h) % patch_h
    pad_w = (patch_w - W % patch_w) % patch_w
    if pad_h > 0 or pad_w > 0:
        images = F.pad(input=images, pad=(0, pad_w, 0, pad_h), mode='constant', value=0)
        B, C, H, W = images.shape

    # Extract patches of size patch_h x patch_w along height and width
    patches = images.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)

    # Calculate number of patches in each dimension
    n_patches_h = H // patch_h
    n_patches_w = W // patch_w

    # Permute from (B, C, n_patches_h, n_patches_w, patch_h, patch_w) to (B, n_patches_h, n_patches_w, C, patch_h, patch_w)
    # Reshape to (B, n_patches_h * n_patches_w, C * patch_h * patch_w)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, n_patches_h * n_patches_w, -1)

    return patches

def positional_embedding_2d(h: int, w: int, dim: int, temperature: int = 10000) -> torch.Tensor:
    """Generates positional embedding for 2D grid of size (h, w) and dimension dim.

    Args:
        h (int): Height of the grid
        w (int): Width of the grid
        dim (int): Dimension of the positional embedding
        temperature (int, optional): Temperature scaling. Defaults to 10000.
    
    Returns:
        torch.Tensor: Positional embedding of shape (h * w, dim)
    """
    # x and y indexes of patch grid
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    assert dim % 4 == 0, 'dim must be divisible by 4 for sin/cos positional embedding'
    
    # calculate omega values
    omega = 1.0 / torch.pow(temperature, torch.arange(dim // 4) / (dim // 4 - 1))
    
    # calculate sin and cos positional embedding
    y = y.flatten().unsqueeze_(-1) * omega.unsqueeze(0)
    x = x.flatten().unsqueeze_(-1) * omega.unsqueeze(0)
    pos = torch.cat((torch.sin(y), torch.cos(y), torch.sin(x), torch.cos(x)), dim=1)

    return pos

