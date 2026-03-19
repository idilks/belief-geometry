"""
Small causal transformer for next-token prediction on Mess3 sequences.

Decoder-only, 3-token vocabulary, designed to be small enough for CPU training
while having enough capacity to learn the component-specific dynamics.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x, return_attention=False, ablate_heads=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att_weights = att  # (B, H, T, T) before dropout
        att = self.dropout(att)

        # per-head outputs before projection: (B, H, T, head_dim)
        head_out = att @ v
        if ablate_heads is not None:
            for h in ablate_heads:
                head_out[:, h] = 0.0

        out = head_out.transpose(1, 2).reshape(B, T, C)
        result = self.out_proj(out)
        if return_attention:
            return result, att_weights
        return result


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False, ablate_heads=None, ablate_mlp=False):
        if return_attention:
            attn_out, att_weights = self.attn(self.ln1(x), return_attention=True, ablate_heads=ablate_heads)
        else:
            attn_out = self.attn(self.ln1(x), ablate_heads=ablate_heads)
            att_weights = None
        x = x + attn_out
        if not ablate_mlp:
            x = x + self.mlp(self.ln2(x))
        if return_attention:
            return x, att_weights
        return x


class Mess3Transformer(nn.Module):
    """
    Tiny decoder-only transformer for 3-token next-token prediction.

    Architecture:
        token embedding + learned positional embedding
        → N transformer blocks (pre-norm)
        → final LayerNorm → linear head to vocab

    The residual stream at each layer is accessible via extract_residual_stream().
    """

    def __init__(
        self,
        vocab_size=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        max_len=15,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_len = max_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, max_len, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying: share token embedding with output head
        self.head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: (B, T) int tensor of token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.shape
        assert T <= self.max_len, f"Sequence length {T} exceeds max_len {self.max_len}"

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        return self.head(h)

    @torch.no_grad()
    def extract_attention_weights(self, x):
        """
        Run forward pass and return attention weight matrices for each layer.

        Returns:
            attention_weights: list of (B, H, T, T) tensors, one per layer
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        all_att = []
        for block in self.blocks:
            h, att = block(h, return_attention=True)
            all_att.append(att)
        return all_att

    @torch.no_grad()
    def forward_with_ablation(self, x, ablate_heads=None, ablate_mlps=None):
        """
        Forward pass with selective ablation of heads and/or MLPs.

        Args:
            x: (B, T) int tensor
            ablate_heads: dict mapping layer_idx -> list of head indices to zero out
            ablate_mlps: list of layer indices whose MLP outputs to zero out
        Returns:
            logits: (B, T, vocab_size)
            residuals: dict with 'final' key -> (B, T, d_model)
        """
        if ablate_heads is None:
            ablate_heads = {}
        if ablate_mlps is None:
            ablate_mlps = []

        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        for i, block in enumerate(self.blocks):
            h = block(h,
                      ablate_heads=ablate_heads.get(i),
                      ablate_mlp=(i in ablate_mlps))

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, h

    @torch.no_grad()
    def extract_residual_stream(self, x):
        """
        Run forward pass and collect the residual stream after each layer.

        Returns:
            residuals: dict mapping layer name to (B, T, d_model) tensor
                'embed'   — after token + position embedding
                'layer_0' — after first transformer block
                'layer_1' — after second transformer block
                ...
                'final'   — after final layer norm (pre-head)
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        residuals = {"embed": h.clone()}

        for i, block in enumerate(self.blocks):
            h = block(h)
            residuals[f"layer_{i}"] = h.clone()

        h = self.ln_f(h)
        residuals["final"] = h.clone()

        return residuals
