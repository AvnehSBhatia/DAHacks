"""
Train encoder/decoder on real MiniLM-L6-v2 text embeddings (384-d).

Encoder input: masked pair only, shape [B, 2, 384] -> flat [B, 768] (2×384).
Drop positions are set to 0; training still uses a hidden mask only for the loss
(MAE on dropped coordinates so the model learns to inpaint).

Run from repo root:  python training/pair_autoencoder.py pregenerate
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterator

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.device import select_torch_device
from models.paths import CHECKPOINTS_DIR, ensure_checkpoints_dir

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from models.sentence_transformer_loader import load_sentence_transformer

# Official sentence-transformers MiniLM v2 — 384-d embeddings
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ensure_checkpoints_dir()
DEFAULT_EMBEDDINGS_PATH = CHECKPOINTS_DIR / "minilm_pair_embeddings.pt"

DIM = 384
LATENT = 64
ENC_IN = 2 * DIM  # flattened [2, 384] -> 768


class Encoder2x384To64(nn.Module):
    """Maps masked [B, 2, 384] (flattened to [B, 768]) -> [B, 64]."""

    def __init__(self, d: int = DIM, latent: int = LATENT) -> None:
        super().__init__()
        self.d = d
        self.latent = latent
        self.fc = nn.Linear(ENC_IN, latent)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        return self.fc(x_masked.reshape(x_masked.size(0), -1))


class Decoder64To2x384(nn.Module):
    """Maps [B, 64] -> [B, 2, 384]."""

    def __init__(self, d: int = DIM, latent: int = LATENT) -> None:
        super().__init__()
        self.d = d
        self.latent = latent
        self.fc = nn.Linear(latent, 2 * d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fc(z)
        return out.view(z.size(0), 2, self.d)


def load_text_lines(min_chars: int = 40, max_lines: int = 50_000) -> list[str]:
    """Pull text lines from WikiText-2 (raw); only real text, no synthetic vectors."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    lines: list[str] = []
    for row in ds:
        t = row["text"].strip()
        if len(t) >= min_chars:
            lines.append(t)
        if len(lines) >= max_lines:
            break
    if len(lines) < 100:
        raise RuntimeError("Too few text lines; check dataset load.")
    return lines


def build_embedding_pool(
    model: SentenceTransformer,
    lines: list[str],
    num_pairs: int,
    encode_batch_size: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    """Encode random (prompt, context) pairs with MiniLM -> [N, 2, 384], L2-normalized like ST."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    for _ in range(num_pairs):
        a, b = rng.sample(lines, 2)
        pairs.append((a, b))

    out_list: list[torch.Tensor] = []
    model.eval()
    batch_starts = range(0, len(pairs), encode_batch_size)
    with torch.no_grad():
        for i in tqdm(
            batch_starts,
            desc="Encoding (prompt, context) pairs",
            unit="batch",
            leave=True,
        ):
            batch = pairs[i : i + encode_batch_size]
            prompts = [p[0] for p in batch]
            contexts = [p[1] for p in batch]
            ep = model.encode(
                prompts,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            ec = model.encode(
                contexts,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            x = torch.stack([ep, ec], dim=1).to(dtype=torch.float32)
            out_list.append(x.cpu())
    return torch.cat(out_list, dim=0)


def pregenerate_embeddings(
    out_path: Path | str,
    num_pairs: int,
    encode_batch_size: int,
    pair_seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Run MiniLM on text pairs and save `[N, 2, 384]` to disk (CPU tensor)."""
    out_path = Path(out_path)
    print("Loading MiniLM-L6-v2 …")
    st = load_sentence_transformer(MINILM_MODEL, str(device))
    print("Loading text (WikiText-2) …")
    lines = load_text_lines()
    print(f"lines: {len(lines)}")
    print(f"Pregenerating {num_pairs} (prompt, context) pairs …")
    pool = build_embedding_pool(
        st, lines, num_pairs=num_pairs, encode_batch_size=encode_batch_size, device=device, seed=pair_seed
    )
    payload = {
        "embeddings": pool,
        "model": MINILM_MODEL,
        "num_pairs": num_pairs,
        "pair_seed": pair_seed,
        "shape": tuple(pool.shape),
    }
    torch.save(payload, out_path)
    print(f"saved {out_path} shape={tuple(pool.shape)}")
    return pool


def load_embedding_pool(path: Path | str) -> torch.Tensor:
    """Load pregenerated `[N, 2, 384]` tensor from `pregenerate_embeddings` output."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Embeddings file not found: {path}. Run: python training/pair_autoencoder.py pregenerate"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    emb = data["embeddings"]
    if not isinstance(emb, torch.Tensor):
        raise TypeError("Expected payload key 'embeddings' to be a Tensor")
    return emb


def random_dropout_mask(
    x: torch.Tensor,
    drop_prob: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B, 2, d] full embeddings.
    Returns (x_masked, mask_drop) where mask_drop is float {0,1}, 1 = dropped coordinate.
    Ensures each sample has at least one dropped value when drop_prob > 0.
    """
    if generator is None:
        mask_drop = torch.rand(x.shape, device=x.device, dtype=x.dtype) < drop_prob
    else:
        mask_drop = torch.rand(x.shape, generator=generator, device=x.device, dtype=x.dtype) < drop_prob
    mask_drop = mask_drop.to(dtype=torch.float32)
    # Guarantee at least one masked dim per sample
    flat = mask_drop.view(x.size(0), -1)
    for i in range(x.size(0)):
        if flat[i].sum() < 1:
            j = torch.randint(0, flat.size(1), (1,), device=x.device, generator=generator).item()
            flat[i, j] = 1.0
    mask_drop = flat.view_as(x)
    x_masked = x * (1.0 - mask_drop)
    return x_masked, mask_drop


def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask_drop: torch.Tensor) -> torch.Tensor:
    """MAE only on dropped positions."""
    err = (pred - target).abs() * mask_drop
    denom = mask_drop.sum().clamp_min(1.0)
    return err.sum() / denom


def iter_train_batches(
    pool: torch.Tensor,
    batch_size: int,
    device: torch.device,
    drop_prob: float,
    seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yields (x_full, x_masked, mask_drop) per batch; x_full for loss target."""
    n = pool.size(0)
    g_perm = torch.Generator()
    g_perm.manual_seed(seed)
    perm = torch.randperm(n, generator=g_perm)
    g_mask = torch.Generator(device=device)
    g_mask.manual_seed(seed + 1337)
    idx = 0
    while idx + batch_size <= n:
        ix = perm[idx : idx + batch_size]
        idx += batch_size
        x = pool[ix].to(device=device, dtype=torch.float32)
        x_masked, mask_drop = random_dropout_mask(x, drop_prob, generator=g_mask)
        yield x, x_masked, mask_drop


def train(
    pool: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    drop_prob: float,
    seed: int,
    device: str | None = None,
) -> tuple[Encoder2x384To64, Decoder64To2x384]:
    torch.manual_seed(seed)
    random.seed(seed)
    dev = select_torch_device(device)
    pool = pool.to(dev)

    encoder = Encoder2x384To64(DIM, LATENT).to(dev)
    decoder = Decoder64To2x384(DIM, LATENT).to(dev)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-4)

    encoder.train()
    decoder.train()
    step = 0
    epoch = 0
    while step < steps:
        for x, x_masked, mask_drop in iter_train_batches(pool, batch_size, dev, drop_prob, seed + epoch):
            if step >= steps:
                break
            opt.zero_grad(set_to_none=True)
            z = encoder(x_masked)
            x_hat = decoder(z)
            loss = masked_mae_loss(x_hat, x, mask_drop)
            loss.backward()
            opt.step()
            step += 1
            if step == 1 or step % 200 == 0:
                print(f"step {step:5d}  masked_mae {loss.item():.6f}")
        epoch += 1

    return encoder, decoder


def _save_encoder_decoder(enc: Encoder2x384To64, dec: Decoder64To2x384) -> None:
    enc_path = CHECKPOINTS_DIR / "encoder_2x384_to_64.pt"
    dec_path = CHECKPOINTS_DIR / "decoder_64_to_2x384.pt"
    torch.save(
        {
            "state_dict": enc.state_dict(),
            "dim": DIM,
            "latent": LATENT,
            "minilm_model": MINILM_MODEL,
            "encoder_in_dim": ENC_IN,
            "arch": "Encoder2x384To64",
        },
        enc_path,
    )
    torch.save(
        {
            "state_dict": dec.state_dict(),
            "dim": DIM,
            "latent": LATENT,
            "minilm_model": MINILM_MODEL,
            "arch": "Decoder64To2x384",
        },
        dec_path,
    )
    print(f"saved {enc_path}")
    print(f"saved {dec_path}")


if __name__ == "__main__":
    dev = select_torch_device()
    parser = argparse.ArgumentParser(description="MiniLM pair embeddings: pregenerate and/or train AE")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("pregenerate", help="Encode text pairs with MiniLM and save embeddings (no training)")
    p_pre.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Output .pt file (default: minilm_pair_embeddings.pt)",
    )
    p_pre.add_argument("--num-pairs", type=int, default=12_000)
    p_pre.add_argument("--encode-batch-size", type=int, default=64)
    p_pre.add_argument("--pair-seed", type=int, default=0, help="RNG seed for sampling text pairs")

    p_tr = sub.add_parser("train", help="Train encoder/decoder from a pregenerated embeddings file")
    p_tr.add_argument(
        "--embeddings",
        type=Path,
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path from pregenerate (default: minilm_pair_embeddings.pt)",
    )
    p_tr.add_argument("--steps", type=int, default=2000)
    p_tr.add_argument("--batch-size", type=int, default=128)
    p_tr.add_argument("--lr", type=float, default=3e-3)
    p_tr.add_argument("--drop-prob", type=float, default=0.06)
    p_tr.add_argument("--seed", type=int, default=0)

    p_all = sub.add_parser("all", help="Pregenerate embeddings then train (one shot)")
    p_all.add_argument("--out", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    p_all.add_argument("--num-pairs", type=int, default=12_000)
    p_all.add_argument("--encode-batch-size", type=int, default=64)
    p_all.add_argument("--pair-seed", type=int, default=0)
    p_all.add_argument("--steps", type=int, default=2000)
    p_all.add_argument("--batch-size", type=int, default=128)
    p_all.add_argument("--lr", type=float, default=3e-3)
    p_all.add_argument("--drop-prob", type=float, default=0.06)
    p_all.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.command == "pregenerate":
        pregenerate_embeddings(
            args.out,
            num_pairs=args.num_pairs,
            encode_batch_size=args.encode_batch_size,
            pair_seed=args.pair_seed,
            device=dev,
        )
    elif args.command == "train":
        print(f"Loading embeddings from {args.embeddings} …")
        pool = load_embedding_pool(args.embeddings)
        print(f"pool shape: {tuple(pool.shape)}")
        enc, dec = train(
            pool,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            drop_prob=args.drop_prob,
            seed=args.seed,
            device=str(dev),
        )
        _save_encoder_decoder(enc, dec)
    else:
        pregenerate_embeddings(
            args.out,
            num_pairs=args.num_pairs,
            encode_batch_size=args.encode_batch_size,
            pair_seed=args.pair_seed,
            device=dev,
        )
        pool = load_embedding_pool(args.out)
        enc, dec = train(
            pool,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            drop_prob=args.drop_prob,
            seed=args.seed,
            device=str(dev),
        )
        _save_encoder_decoder(enc, dec)
