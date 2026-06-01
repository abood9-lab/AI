"""
VexoLM Training Script (Streaming, Gradient Accumulation, Checkpointing)
========================================================================

This script trains the core decoder-only text model (VexoLMDecoder) using:
- Hugging Face `datasets` with streaming=True (iterates without loading into RAM)
- CustomTokenizer from model.py (bilingual Arabic/English)
- PyTorch training loop with:
  * gradient accumulation
  * AdamW optimizer
  * warmup + cosine LR scheduler
  * periodic checkpoint saving (model + optimizer + scheduler + vocab)

Designed for cloud environments (GitHub Codespaces/Actions):
- Works with limited RAM via streaming datasets
- Uses accumulation to simulate larger batch sizes
- Includes clear logging for loss tracking

Usage example:
--------------
python train.py \
  --dataset 2A2I/Arabic-OpenHermes-2.5 \
  --split train \
  --text_field text \
  --max_steps 2000 \
  --seq_len 512 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --lr 3e-4 \
  --warmup_steps 200 \
  --save_every 500

Notes:
------
- For streaming datasets, we can't reliably precompute dataset length. We train by `max_steps`.
- This script focuses on text-only training. Vision integration can be added later.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import asdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch
from torch.optim import AdamW

try:
    from datasets import load_dataset
except ImportError as e:
    raise ImportError("Please install datasets: pip install datasets") from e

from model import CustomTokenizer, VexoLMConfig, VexoLMDecoder


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: str,
    *,
    model: VexoLMDecoder,
    optimizer: torch.optim.Optimizer,
    scheduler_state: Dict,
    step: int,
    cfg: VexoLMConfig,
    tokenizer: CustomTokenizer,
) -> None:
    """Save a checkpoint atomically."""
    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": scheduler_state,
        "config": asdict(cfg),
        "tokenizer": {
            "token_to_id": tokenizer.token_to_id,
            "lowercase_english": tokenizer.lowercase_english,
        },
    }
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)
    print(f"[checkpoint] Saved: {path} (step={step})")


def load_checkpoint(path: str, *, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)


# =============================================================================
# Streaming batching + tokenization
# =============================================================================

def iter_text_samples(streaming_dataset: Iterable[Dict], text_field: str) -> Iterator[str]:
    for row in streaming_dataset:
        if text_field not in row:
            raise KeyError(
                f"Field '{text_field}' not found. Available keys: {list(row.keys())}"
            )
        text = row[text_field]
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            continue
        yield text


def build_vocab_from_stream(
    tokenizer: CustomTokenizer,
    text_iter: Iterator[str],
    *,
    vocab_build_steps: int,
    min_freq: int,
    max_vocab_size: int,
) -> None:
    """Build vocabulary from first N samples (consumes iterator)."""
    texts: List[str] = []
    for i, t in enumerate(text_iter):
        texts.append(t)
        if i + 1 >= vocab_build_steps:
            break

    if not texts:
        raise RuntimeError("No texts found while building vocabulary.")

    print(f"[vocab] Building vocab from {len(texts)} samples...")
    tokenizer.build_vocab(texts, min_freq=min_freq, max_vocab_size=max_vocab_size)
    print(f"[vocab] Vocab size = {len(tokenizer)}")


def make_token_id_stream(
    tokenizer: CustomTokenizer,
    text_stream: Iterator[str],
    *,
    add_bos: bool = True,
    add_eos: bool = True,
) -> Iterator[List[int]]:
    for text in text_stream:
        yield tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)


def pack_tokens_to_fixed_length(
    token_ids_iter: Iterator[List[int]],
    *,
    seq_len: int,
) -> Iterator[List[int]]:
    """Pack variable-length sequences into fixed chunks of length seq_len+1."""
    buffer: List[int] = []
    needed = seq_len + 1

    for ids in token_ids_iter:
        if not ids:
            continue
        buffer.extend(ids)

        while len(buffer) >= needed:
            chunk = buffer[:needed]
            buffer = buffer[needed:]
            yield chunk


def batch_fixed_sequences(
    fixed_seq_iter: Iterator[List[int]],
    *,
    batch_size: int,
) -> Iterator[List[List[int]]]:
    batch: List[List[int]] = []
    for seq in fixed_seq_iter:
        batch.append(seq)
        if len(batch) >= batch_size:
            yield batch
            batch = []


def collate_batch(
    batch: List[List[int]],
    *,
    device: torch.device,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert batch of chunks (seq_len+1) into (input_ids, targets, attention_mask)."""
    b = len(batch)
    input_ids = torch.empty((b, seq_len), dtype=torch.long, device=device)
    targets = torch.empty((b, seq_len), dtype=torch.long, device=device)

    for i, chunk in enumerate(batch):
        if len(chunk) != seq_len + 1:
            raise ValueError("Internal error: chunk is not seq_len+1")
        input_ids[i] = torch.tensor(chunk[:-1], dtype=torch.long, device=device)
        targets[i] = torch.tensor(chunk[1:], dtype=torch.long, device=device)

    attention_mask = torch.ones((b, seq_len), dtype=torch.long, device=device)
    return input_ids, targets, attention_mask


# =============================================================================
# Learning rate schedule (warmup + cosine)
# =============================================================================

class WarmupCosineScheduler:
    """Simple warmup + cosine decay LR scheduler with save/restore state."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr = float(min_lr)

        self.step_num = 0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self) -> float:
        t = self.step_num

        if self.warmup_steps > 0 and t < self.warmup_steps:
            return self.base_lr * (t + 1) / self.warmup_steps

        if self.total_steps <= self.warmup_steps:
            return self.base_lr

        progress = (t - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine

    def step(self) -> float:
        lr = float(self.get_lr())
        self._set_lr(lr)
        self.step_num += 1
        return lr

    def state_dict(self) -> Dict:
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state: Dict) -> None:
        self.base_lr = float(state["base_lr"])
        self.warmup_steps = int(state["warmup_steps"])
        self.total_steps = int(state["total_steps"])
        self.min_lr = float(state["min_lr"])
        self.step_num = int(state["step_num"])
        self._set_lr(float(self.get_lr()))


# =============================================================================
# Training
# =============================================================================

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    print(f"[env] device={device} cuda={torch.cuda.is_available()} torch={torch.__version__}")

    # Load streaming dataset
    print(f"[data] Loading dataset={args.dataset} split={args.split} streaming=True ...")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    # Tokenizer + vocab build from prefix
    tokenizer = CustomTokenizer(vocab=None, add_special_tokens=True, lowercase_english=True)

    vocab_stream = iter_text_samples(ds, args.text_field)
    build_vocab_from_stream(
        tokenizer,
        vocab_stream,
        vocab_build_steps=args.vocab_build_steps,
        min_freq=args.vocab_min_freq,
        max_vocab_size=args.vocab_size,
    )

    # Re-open stream for training (since vocab building consumes iterator)
    ds_train = load_dataset(args.dataset, split=args.split, streaming=True)
    text_stream = iter_text_samples(ds_train, args.text_field)
    token_stream = make_token_id_stream(tokenizer, text_stream, add_bos=True, add_eos=True)
    fixed_seq_stream = pack_tokens_to_fixed_length(token_stream, seq_len=args.seq_len)
    batch_stream = batch_fixed_sequences(fixed_seq_stream, batch_size=args.batch_size)

    # Build model
    cfg = VexoLMConfig(
        vocab_size=len(tokenizer),
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        attn_dropout=args.attn_dropout,
        resid_dropout=args.resid_dropout,
        ffn_dropout=args.ffn_dropout,
        use_rope=True,
        tie_weights=True,
    )

    model = VexoLMDecoder(cfg).to(device)

    # Optional freezing
    if args.freeze_embeddings or args.freeze_n_layers > 0:
        model.freeze_layers(freeze_embeddings=args.freeze_embeddings, freeze_n_layers=args.freeze_n_layers)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[freeze] trainable_params={trainable:,} / total_params={total:,}")

    # AMP
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Optimizer + scheduler
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    scheduler = WarmupCosineScheduler(
        optimizer,
        base_lr=args.lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
        min_lr=args.min_lr,
    )

    # Resume
    start_step = 0
    if args.resume and os.path.isfile(args.checkpoint_path):
        print(f"[resume] Loading checkpoint: {args.checkpoint_path}")
        ckpt = load_checkpoint(args.checkpoint_path, device=device)

        tokenizer = CustomTokenizer(
            vocab=ckpt["tokenizer"]["token_to_id"],
            add_special_tokens=True,
            lowercase_english=ckpt["tokenizer"].get("lowercase_english", True),
        )
        cfg = VexoLMConfig(**ckpt["config"])
        model = VexoLMDecoder(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        scheduler = WarmupCosineScheduler(
            optimizer,
            base_lr=float(ckpt["scheduler_state"]["base_lr"]),
            warmup_steps=int(ckpt["scheduler_state"]["warmup_steps"]),
            total_steps=int(ckpt["scheduler_state"]["total_steps"]),
            min_lr=float(ckpt["scheduler_state"]["min_lr"]),
        )
        scheduler.load_state_dict(ckpt["scheduler_state"])

        start_step = int(ckpt["step"])
        print(f"[resume] Resumed from step={start_step}")

        # Rebuild streams
        ds_train = load_dataset(args.dataset, split=args.split, streaming=True)
        text_stream = iter_text_samples(ds_train, args.text_field)
        token_stream = make_token_id_stream(tokenizer, text_stream, add_bos=True, add_eos=True)
        fixed_seq_stream = pack_tokens_to_fixed_length(token_stream, seq_len=args.seq_len)
        batch_stream = batch_fixed_sequences(fixed_seq_stream, batch_size=args.batch_size)

    # Train
    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(
        "[train] starting...\n"
        f"  max_steps={args.max_steps} (optimizer steps)\n"
        f"  batch_size={args.batch_size}\n"
        f"  grad_accum_steps={args.grad_accum_steps}\n"
        f"  seq_len={args.seq_len}\n"
        f"  effective_tokens_per_step={args.batch_size * args.grad_accum_steps * args.seq_len}\n"
        f"  amp={use_amp}\n"
        f"  checkpoint_path={args.checkpoint_path}\n"
    )

    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    global_step = start_step

    while global_step < args.max_steps:
        # Accumulate gradients
        for _micro in range(args.grad_accum_steps):
            try:
                batch = next(batch_stream)
            except StopIteration:
                print("[data] Stream exhausted. Ending training.")
                global_step = args.max_steps
                break

            input_ids, targets, attn_mask = collate_batch(batch, device=device, seq_len=args.seq_len)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids, attention_mask=attn_mask)
                loss = model.compute_loss(logits, targets)
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            running_loss += float(loss.item()) * args.grad_accum_steps
            running_count += 1

        if global_step >= args.max_steps:
            break

        # Clip grads
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        lr = scheduler.step()
        global_step += 1

        # Logging
        if global_step % args.log_every == 0:
            elapsed = time.time() - t0
            avg_loss = running_loss / max(1, running_count)

            tokens_processed = args.log_every * args.batch_size * args.grad_accum_steps * args.seq_len
            tok_s = tokens_processed / max(elapsed, 1e-9)

            print(
                f"[step {global_step:>6}/{args.max_steps}] "
                f"loss={avg_loss:.4f} lr={lr:.6g} "
                f"tok/s≈{tok_s:.0f} elapsed={elapsed:.1f}s"
            )

            t0 = time.time()
            running_loss = 0.0
            running_count = 0

        # Checkpoint
        if (args.save_every > 0 and global_step % args.save_every == 0) or (global_step == args.max_steps):
            save_checkpoint(
                args.checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler_state=scheduler.state_dict(),
                step=global_step,
                cfg=cfg,
                tokenizer=tokenizer,
            )

    # Always save final
    save_checkpoint(
        args.checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler_state=scheduler.state_dict(),
        step=global_step,
        cfg=cfg,
        tokenizer=tokenizer,
    )

    print("[train] done.")


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train VexoLM core text model (streaming + grad accumulation).")

    # Dataset
    p.add_argument("--dataset", type=str, default="2A2I/Arabic-OpenHermes-2.5")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--text_field", type=str, default="text")

    # Vocab
    p.add_argument("--vocab_build_steps", type=int, default=50_000)
    p.add_argument("--vocab_min_freq", type=int, default=2)
    p.add_argument("--vocab_size", type=int, default=80_000)

    # Training
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=1024)

    # Model
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=8)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--ffn_hidden_dim", type=int, default=1536)

    # Dropouts
    p.add_argument("--attn_dropout", type=float, default=0.0)
    p.add_argument("--resid_dropout", type=float, default=0.0)
    p.add_argument("--ffn_dropout", type=float, default=0.0)

    # Optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Perf
    p.add_argument("--amp", action="store_true")

    # Freezing
    p.add_argument("--freeze_embeddings", action="store_true")
    p.add_argument("--freeze_n_layers", type=int, default=0)

    # Checkpoint
    p.add_argument("--checkpoint_path", type=str, default="vexolm_checkpoint.pth")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--resume", action="store_true")

    # Logging
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
