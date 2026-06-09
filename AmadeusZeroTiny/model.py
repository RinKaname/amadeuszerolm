import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math

# 1. CONFIG
@dataclass
class Config:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int = 2
    n_embd: int = 384
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

# 2. RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.scale / (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()

# 3. RoPE precompute
def precompute_rope_freqs(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

# 4. RoPE apply
def apply_rotary_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)

    head_dim = q.shape[-1]
    q_real, q_imag = q[..., :head_dim//2], q[..., head_dim//2:]
    k_real, k_imag = k[..., :head_dim//2], k[..., head_dim//2:]

    q_rot = torch.cat((q_real * cos - q_imag * sin, q_real * sin + q_imag * cos), dim=-1)
    k_rot = torch.cat((k_real * cos - k_imag * sin, k_real * sin + k_imag * cos), dim=-1)
    return q_rot, k_rot

# 5. BLOCK
class AmadeusZeroTinyBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        assert self.n_head % self.n_kv_head == 0
        self.num_key_value_groups = self.n_head // self.n_kv_head

        hidden_dim = int(8 * config.n_embd / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)

        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, self.q_size + 2 * self.kv_size, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.mlp = nn.ModuleDict({
            'gate_proj': nn.Linear(config.n_embd, hidden_dim, bias=False),
            'up_proj': nn.Linear(config.n_embd, hidden_dim, bias=False),
            'down_proj': nn.Linear(hidden_dim, config.n_embd, bias=False),
        })

    def forward(self, x, cos, sin, position_ids):
        x = x + self._attn_block(self.ln_1(x), cos, sin, position_ids)
        x = x + self._mlp_block(self.ln_2(x))
        return x

    def _attn_block(self, x, cos, sin, position_ids):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=2)

        # Shape: [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        # Shape: [B, T, n_kv_head, head_dim] -> [B, n_kv_head, T, head_dim]
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, cos, sin, position_ids)

        # --------------------------------------------------------
        # OPTIMIZATION FOR RTX 3060: Vectorized GQA
        # Instead of a Python for-loop which is slow and launches
        # multiple CUDA kernels, we use repeat_interleave to duplicate
        # the KV heads. This allows a single scaled_dot_product_attention
        # call, which PyTorch will automatically route to ultra-fast
        # FlashAttention kernels on Ampere GPUs.
        # --------------------------------------------------------
        if self.n_kv_head < self.n_head:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # --------------------------------------------------------

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

    def _mlp_block(self, x):
        gate = F.silu(self.mlp.gate_proj(x))
        up = self.mlp.up_proj(x)
        return self.mlp.down_proj(gate * up)

# 6. MAIN MODEL
class AmadeusZeroTiny(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            'h': nn.ModuleList([AmadeusZeroTinyBlock(config) for _ in range(config.n_layer)]),
            'ln_f': RMSNorm(config.n_embd, eps=config.norm_eps),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

        dim = config.n_embd // config.n_head
        max_len = config.block_size * 2
        cos, sin = precompute_rope_freqs(dim, max_len, theta=config.rope_theta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if module.weight.size(0) == self.config.n_embd:
                std = 0.02 / math.sqrt(2 * self.config.n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, position_ids=None):
        _, t = idx.size()
        assert t <= self.config.block_size

        if position_ids is None:
            position_ids = torch.arange(t, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)

        x = tok_emb
        for block in self.transformer.h:
            x = block(x, self.cos, self.sin, position_ids)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
