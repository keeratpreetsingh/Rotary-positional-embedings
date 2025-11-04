import torch
import torch.nn as nn 
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat([freqs.sin(), freqs.cos()], dim=-1)
        emb = emb.unsqueeze(0).expand(x.size(0), -1, -1)
        x1, x2 = x[..., :self.dim//2], x[..., self.dim//2:]
        sin, cos = emb[..., :self.dim//2], emb[..., self.dim//2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
