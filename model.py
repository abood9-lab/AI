"""
VexoLM - Core Text Architecture (Decoder-only Transformer) + Multimodal (Vision->Text) adapter
===========================================================================================

This file contains the "core text" module for a custom multimodal language model built
from scratch using PyTorch. It is designed to be:

- Production-ready: clear structure, robust checks, no placeholders.
- Modular: components are reusable and easy to swap (attention, FFN, norms, embeddings).
- Multilingual-friendly: a simple custom tokenizer that supports Arabic + English.
- Multimodal-ready: an interface layer that projects vision embeddings into the text
  embedding space so text+image tokens can be processed together.

Notes:
- This file focuses on the **text backbone** and the **vision-to-text projection interface**.
- The tokenizer here is intentionally "custom" and self-contained (no external tokenizers).
  For best quality in real training, you might later replace it with a trained BPE/SentencePiece.
- The model uses **pre-norm** Transformer blocks (LayerNorm before attention/FFN) for stability.
- Uses **causal masking** for GPT-like decoder-only behavior.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Tokenizer
# =============================================================================

class CustomTokenizer:
    """A lightweight custom tokenizer for bilingual Arabic/English text.

    Goals:
    - Deterministic tokenization without external dependencies.
    - Handle Arabic + English + digits + punctuation.
    - Support special tokens: <pad>, <bos>, <eos>, <unk>

    Approach:
    - Normalize unicode (NFKC) to reduce variant forms.
    - Split text using a regex capturing:
      * Arabic word chunks
      * English word chunks (simple apostrophe support)
      * numbers
      * newlines as explicit tokens
      * all other non-whitespace single characters as tokens

    Vocabulary:
    - Build from corpus prefix via `build_vocab(...)`.
    - Unknown tokens map to <unk>.
    """

    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        *,
        add_special_tokens: bool = True,
        lowercase_english: bool = True,
    ) -> None:
        self.lowercase_english = lowercase_english

        if vocab is None:
            vocab = {}

        self.token_to_id: Dict[str, int] = dict(vocab)
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}

        if add_special_tokens:
            for tok in (self.PAD, self.BOS, self.EOS, self.UNK):
                self._ensure_token(tok)

        # Cache special ids
        self.pad_id = self.token_to_id[self.PAD]
        self.bos_id = self.token_to_id[self.BOS]
        self.eos_id = self.token_to_id[self.EOS]
        self.unk_id = self.token_to_id[self.UNK]

        self._token_pattern = re.compile(
            r"(\n)"
            r"|([\u0600-\u06FF]+)"                 # Arabic word chunk
            r"|([A-Za-z]+(?:'[A-Za-z]+)?)"         # English word chunk
            r"|(\d+)"                               # number
            r"|([^\s])",                            # any other single non-space char
            flags=re.UNICODE
        )

    def _ensure_token(self, token: str) -> int:
        if token in self.token_to_id:
            return self.token_to_id[token]
        idx = len(self.token_to_id)
        self.token_to_id[token] = idx
        self.id_to_token[idx] = token
        return idx

    @staticmethod
    def _normalize(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    def tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        if self.lowercase_english:
            text = text.lower()

        tokens: List[str] = []
        for match in self._token_pattern.finditer(text):
            tok = next(g for g in match.groups() if g is not None)
            tokens.append(tok)
        return tokens

    @staticmethod
    def _is_punct(s: str) -> bool:
        if s == "\n":
            return False
        if len(s) == 1 and not s.isalnum():
            return True
        return False

    def detokenize(self, tokens: Sequence[str]) -> str:
        out: List[str] = []
        for tok in tokens:
            if tok in (self.PAD, self.BOS, self.EOS):
                continue
            if tok == "\n":
                out.append("\n")
                continue

            if not out:
                out.append(tok)
                continue

            prev = out[-1]
            if prev.endswith("\n"):
                out.append(tok)
            elif self._is_punct(tok):
                out.append(tok)
            elif self._is_punct(prev[-1]):
                out.append(" " + tok)
            else:
                out.append(" " + tok)

        return "".join(out)

    def build_vocab(
        self,
        texts: Iterable[str],
        *,
        min_freq: int = 2,
        max_vocab_size: Optional[int] = None,
    ) -> None:
        freq: Dict[str, int] = {}
        for t in texts:
            for tok in self.tokenize(t):
                freq[tok] = freq.get(tok, 0) + 1

        items = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for tok, f in items:
            if f < min_freq:
                continue
            if tok in self.token_to_id:
                continue
            if max_vocab_size is not None and len(self.token_to_id) >= max_vocab_size:
                break
            self._ensure_token(tok)

        # refresh cached ids
        self.pad_id = self.token_to_id[self.PAD]
        self.bos_id = self.token_to_id[self.BOS]
        self.eos_id = self.token_to_id[self.EOS]
        self.unk_id = self.token_to_id[self.UNK]

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        tokens = [self.id_to_token.get(int(i), self.UNK) for i in ids]
        return self.detokenize(tokens)

    def __len__(self) -> int:
        return len(self.token_to_id)


# =============================================================================
# Core Transformer building blocks
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) applied to Q/K in attention."""

    def __init__(self, head_dim: int, base: int = 10000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached: Dict[Tuple[torch.device, torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_cos_sin(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype, seq_len)
        if key in self._cached:
            return self._cached[key]

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, head_dim/2]
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        self._cached[key] = (cos, sin)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, h, s, d = q.shape
        if d != self.head_dim:
            raise ValueError(f"Expected head_dim {self.head_dim}, got {d}")

        cos, sin = self._get_cos_sin(seq_len=s, device=q.device, dtype=q.dtype)
        cos_full = torch.repeat_interleave(cos.unsqueeze(0).unsqueeze(0), repeats=2, dim=-1)
        sin_full = torch.repeat_interleave(sin.unsqueeze(0).unsqueeze(0), repeats=2, dim=-1)

        q_out = (q * cos_full) + (self._rotate_half(q) * sin_full)
        k_out = (k * cos_full) + (self._rotate_half(k) * sin_full)
        return q_out, k_out


class MultiHeadSelfAttention(nn.Module):
    """GPT-style multi-head self-attention with causal masking and optional RoPE."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbedding(self.head_dim) if use_rope else None

    def forward(self, x: torch.Tensor, *, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, e = x.shape
        if e != self.embed_dim:
            raise ValueError(f"Expected embed_dim {self.embed_dim}, got {e}")

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and self.rope is not None:
            q, k = self.rope.apply_rotary(q, k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal, float("-inf"))

        if attention_mask is not None:
            if attention_mask.shape != (b, s):
                raise ValueError(f"attention_mask must be [batch, seq_len] = {(b, s)}")
            key_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [b,1,1,s]
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        y = torch.matmul(attn_probs, v)  # [b, heads, s, head_dim]
        y = y.transpose(1, 2).contiguous().view(b, s, e)

        y = self.out(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    """Gated FFN (SwiGLU-style) for strong performance."""

    def __init__(self, embed_dim: int, hidden_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc_in = nn.Linear(embed_dim, 2 * hidden_dim, bias=False)
        self.fc_out = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.fc_in(x)
        gate, value = x_in.chunk(2, dim=-1)
        x_act = F.silu(gate) * value
        x_out = self.fc_out(x_act)
        return self.dropout(x_out)


class TransformerBlock(nn.Module):
    """Pre-norm decoder-only Transformer block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        *,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            use_rope=use_rope,
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim=embed_dim, hidden_dim=ffn_hidden_dim, dropout=ffn_dropout)

    def forward(self, x: torch.Tensor, *, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


# =============================================================================
# Vision Encoder Connection Interface (Projection Layer)
# =============================================================================

class VisionToTextProjector(nn.Module):
    """Projects vision embeddings [b, n_patches, vision_dim] into text embedding space."""

    def __init__(
        self,
        vision_dim: int,
        text_embed_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.text_embed_dim = text_embed_dim
        self.hidden_dim = hidden_dim

        self.ln = nn.LayerNorm(vision_dim)
        if hidden_dim is None:
            self.proj = nn.Linear(vision_dim, text_embed_dim, bias=False)
            self.mlp = None
        else:
            self.proj = nn.Linear(vision_dim, hidden_dim, bias=False)
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_dim, text_embed_dim, bias=False),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        if vision_embeds.dim() != 3:
            raise ValueError("vision_embeds must have shape [batch, num_patches, vision_dim]")
        b, n, d = vision_embeds.shape
        if d != self.vision_dim:
            raise ValueError(f"Expected vision_dim={self.vision_dim}, got {d}")

        x = self.ln(vision_embeds)
        x = self.proj(x)
        if self.mlp is not None:
            x = self.mlp(x)
        x = self.dropout(x)
        return x


# =============================================================================
# VexoLM Decoder-only Language Model
# =============================================================================

@dataclass(frozen=True)
class VexoLMConfig:
    vocab_size: int
    max_seq_len: int

    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_hidden_dim: int = 2048

    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    ffn_dropout: float = 0.0

    use_rope: bool = True
    tie_weights: bool = True


class VexoLMDecoder(nn.Module):
    """Decoder-only Transformer returning logits over vocabulary."""

    def __init__(self, cfg: VexoLMConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.drop = nn.Dropout(cfg.resid_dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=cfg.embed_dim,
                num_heads=cfg.num_heads,
                ffn_hidden_dim=cfg.ffn_hidden_dim,
                attn_dropout=cfg.attn_dropout,
                resid_dropout=cfg.resid_dropout,
                ffn_dropout=cfg.ffn_dropout,
                use_rope=cfg.use_rope,
            )
            for _ in range(cfg.num_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def freeze_layers(self, *, freeze_embeddings: bool = True, freeze_n_layers: int = 0) -> None:
        if freeze_embeddings:
            for p in self.tok_emb.parameters():
                p.requires_grad = False

        if freeze_n_layers > 0:
            freeze_n_layers = min(freeze_n_layers, len(self.blocks))
            for i in range(freeze_n_layers):
                for p in self.blocks[i].parameters():
                    p.requires_grad = False

    def _check_seq_len(self, total_seq_len: int) -> None:
        if total_seq_len > self.cfg.max_seq_len:
            raise ValueError(
                f"Sequence length {total_seq_len} exceeds max_seq_len={self.cfg.max_seq_len}. "
                "Increase max_seq_len or shorten inputs."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        vision_embeds: Optional[torch.Tensor] = None,
        vision_projector: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        b, s_text = input_ids.shape
        x_text = self.tok_emb(input_ids)

        if vision_embeds is not None:
            if vision_projector is None:
                raise ValueError("vision_projector must be provided when vision_embeds is provided.")
            x_vision = vision_projector(vision_embeds)
            x = torch.cat([x_vision, x_text], dim=1)

            if attention_mask is not None:
                if attention_mask.shape != (b, s_text):
                    raise ValueError(f"attention_mask must be [batch, s_text] = {(b, s_text)}")
                vision_mask = torch.ones((b, x_vision.size(1)), device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            x = x_text

        self._check_seq_len(x.size(1))

        x = self.drop(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, *, ignore_index: int = -100) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError("logits must be [batch, seq_len, vocab_size]")
        if targets.shape != logits.shape[:2]:
            raise ValueError("targets must match logits first two dims: [batch, seq_len]")

        b, s, v = logits.shape
        return F.cross_entropy(logits.view(b * s, v), targets.view(b * s), ignore_index=ignore_index)
