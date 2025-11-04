# Rotary Positional Embedding (RoPE) — PyTorch Implementation






## Author

**Implementation by:** Keeratpreet Singh

**Background:** Self-taught 16-year-old developer exploring efficient Transformer mechanisms and LLM internals.

**Concept originally from:**
Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding (2021)

**Framework:** PyTorch

**Language:** Python

## Disclaimer

This is an independent educational re-implementation of the Rotary Positional Embedding (RoPE) mechanism introduced in the RoFormer paper.
I am not affiliated with Google, DeepSeek, or any related research group.
This project is intended purely for learning, open research, and experimentation.

## Overview

Traditional Transformers rely on absolute positional encodings, which inject position information additively.
Rotary Positional Embedding (RoPE) instead encodes positions through rotations in embedding space, allowing relative position awareness and better generalization to longer sequences.

This repository provides a minimal PyTorch implementation of RoPE, designed to integrate seamlessly into attention layers.

## Key Features

Continuous Rotational Encoding:
Injects position information multiplicatively through sinusoidal rotation.

Relative Position Awareness:
Enables the model to capture relationships between tokens based on their distance, not just absolute indices.

Plug-and-Play Module:
Can be dropped into any attention mechanism by applying RoPE to queries and keys.

Minimal Dependencies:
Written purely in PyTorch — concise and research-friendly.

## Code Overview
### RotaryEmbedding Class
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


Parameters

dim: Embedding dimension (must be even)

base: Frequency scaling constant (default: 10,000)

🧮 Mathematical Intuition

For each token position p and embedding dimension i, the RoPE mechanism defines:

𝜃
𝑖
=
1
base
2
𝑖
𝑑
θ
i
	​

=
base
d
2i
	​

1
	​


and applies a rotation:

RoPE
(
𝑥
𝑝
)
=
[
𝑥
1
cos
⁡
(
𝜃
𝑝
)
−
𝑥
2
sin
⁡
(
𝜃
𝑝
)


𝑥
1
sin
⁡
(
𝜃
𝑝
)
+
𝑥
2
cos
⁡
(
𝜃
𝑝
)
]
RoPE(x
p
	​

)=[
x
1
	​

cos(θ
p
	​

)−x
2
	​

sin(θ
p
	​

)
x
1
	​

sin(θ
p
	​

)+x
2
	​

cos(θ
p
	​

)
	​

]

This effectively rotates each embedding pair by a position-dependent angle, encoding order directly in the vector geometry.

📘 Example Usage
import torch
from rope import RotaryEmbedding

# Example: embedding dimension = 512
x = torch.randn(2, 128, 512)  # (batch, seq_len, dim)

rope = RotaryEmbedding(dim=512)
rotated = rope(x)

print(rotated.shape)  # (2, 128, 512)


In a Transformer:

q = RotaryEmbedding(dim=512)(q)
k = RotaryEmbedding(dim=512)(k)

🔬 Research References

Su et al. (2021) — RoFormer: Enhanced Transformer with Rotary Position Embedding

Press et al. (2021) — Train Short, Test Long: Attention with Linear Biases (ALiBi)

Touvron et al. (2023) — LLaMA: Open and Efficient Foundation Language Models

DeepSeek-V3 (2025) — Latent Attention with Rotary Positional Encoding

🧠 Possible Applications

Integrating positional encoding in custom Transformer models

Replacing absolute encodings in LLM architectures

Studying the geometric effects of rotational embeddings

Educational demos on sequence modeling

📚 Citation

If you reference or use this implementation, please cite both the RoFormer paper and this educational version:

@software{keeratsingh2025rope,
  author = {Keeratpreet Singh},
  title = {Rotary Positional Embedding (RoPE) — PyTorch Implementation},
  year = {2025},
  url = {https://github.com/Keeratpreetsingh}
}

💡 Contact

📧 Email: keeratpreetsingh2@gmail.com

🌐 GitHub: Keeratpreetsingh

🪪 License

This repository is released under the MIT License.
Original concept © 2021 RoFormer / Google Research.
Implementation © 2025 Keeratpreet Singh.
