from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    seq_len: int

    def to_dict(self) -> dict:
        return asdict(self)


def get_position_encoding(seq_len: int, d_model: int, *, device: torch.device, base: int = 10000) -> torch.Tensor:
    position = torch.zeros(seq_len, d_model, device=device)
    for pos in range(seq_len):
        for i in range(0, d_model // 2):
            angle = pos / (base ** ((2 * i) / d_model))
            position[pos, 2 * i] = math.sin(angle)
            if 2 * i + 1 < d_model:
                position[pos, 2 * i + 1] = math.cos(angle)
    return position.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.Wqkv = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        mask = torch.tril(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.seq_len, config.seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape
        qkv = self.Wqkv(x)
        q, k, v = qkv.split(n_embd, dim=2)

        q = q.reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_scores = attention_scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = attention_weights @ v
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, n_embd)
        return self.proj(attention_output)


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ffn = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
        )
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig, device: torch.device) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.register_buffer(
            "position_encoding",
            get_position_encoding(config.seq_len, config.n_embd, device=device),
        )
        self.blocks = nn.Sequential(*[GPTBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(x) + self.position_encoding[:, : x.size(1), :]
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
