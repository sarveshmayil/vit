import torch
import torch.nn.functional as F

def patchify(images: torch.Tensor, n_patches: int) -> torch.Tensor:
    B, C, H, W = images.shape

    # If H != W, reshape images to be square
    if H != W:
        min_dim = min(H, W)
        images = F.interpolate(images, size=(min_dim, min_dim), mode='bilinear', align_corners=False)
        B, C, H, W = images.shape

    pad = H % n_patches
    images = F.pad(input=images, pad=(pad//2, pad-pad//2, pad//2, pad-pad//2), mode='constant', value=0)

    patch_dim = H // n_patches

    # Extract patches of size patch_dim along height and width
    patches = images.unfold(2,patch_dim,patch_dim).unfold(3,patch_dim,patch_dim)

    # Permute from (B, C, n_patches, n_patches, patch_dim, patch_dim) to (B, n_patches, n_patches, C, patch_dim, patch_dim)
    # Reshape to (B, n_patches**2, C*patch_dim*patch_dim)
    patches = patches.permute(0,2,3,1,4,5).contiguous().view(1, n_patches**2, -1)

    return patches
