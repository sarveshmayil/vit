import torch.nn as nn

class MultiHeadAttention(nn.Module):
    r"""MultiHeadAttention module from 'Attention Is All You Need'
    Method described in the paper:
    [Attention Is All You Need](https://arxiv.org/abs/1706.03762).

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    where :math:`\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V`.
    """
    def __init__(self, dim: int, num_heads: int=8, head_dim: int=64):
        super().__init__()
        hidden_dim = head_dim * num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        B, N, C = x.shape

        # Project paches to Q, K, V representations
        # (B,N,C) -> (B,N,3*num_heads*head_dim) -> (B,N,3,num_heads,head_dim) -> (3,B,num_heads,N,head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Perform multi-head self-attention
        # This finds the similarity between each patch and every other patch
        # This is then scaled by the sqrt of the head_dim because the dot product grows with the dimension
        #   This may cause the softmax to saturate in regions with small gradients, so we scale it down
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) -> (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Apply attention to the values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, num_heads*head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # (B, N, num_heads*head_dim) -> (B, N, dim)
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer encoder block.

    Patches -> Layer Norm -> Multi-Head Attention ---> Layer Norm -> Feed Forward -> Out
      └───────────────────────────────────────────┘ └─────────────────────────────┘

    (B,N,C) -> (B,N,C)
    """
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
