"""
Text-embedding autoencoder: MiniLM-L6-v2 (384-d) -> 64 -> 384.

Pregenerate single-sentence embeddings from WikiText-2, then train with MAE.
Saves encoder and decoder as separate checkpoints.

Run:  python training/text_autoencoder.py pregenerate
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.device import select_torch_device
from models.paths import CHECKPOINTS_DIR, ensure_checkpoints_dir

ensure_checkpoints_dir()

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from models.sentence_transformer_loader import load_sentence_transformer

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMB_PATH = CHECKPOINTS_DIR / "minilm_text_embeddings_384.pt"

DIM = 384
LATENT = 64


class Encoder384To64(nn.Module):
    """384-d text embedding -> 64-d latent."""

    def __init__(self, d: int = DIM, latent: int = LATENT) -> None:
        super().__init__()
        self.d = d
        self.latent = latent
        self.fc = nn.Linear(d, latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Decoder64To384(nn.Module):
    """64-d latent -> 384-d reconstruction."""

    def __init__(self, d: int = DIM, latent: int = LATENT) -> None:
        super().__init__()
        self.d = d
        self.latent = latent
        self.fc = nn.Linear(latent, d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


def load_text_lines(min_chars: int = 40, max_lines: int = 50_000) -> list[str]:
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


def build_text_embedding_matrix(
    model: SentenceTransformer,
    lines: list[str],
    num_texts: int,
    encode_batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Sample `num_texts` lines (with replacement of pool) and encode -> [N, 384], L2-normalized."""
    rng = random.Random(seed)
    texts = [rng.choice(lines) for _ in range(num_texts)]
    out_list: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(
            range(0, len(texts), encode_batch_size),
            desc="Encoding texts (MiniLM)",
            unit="batch",
        ):
            batch = texts[i : i + encode_batch_size]
            e = model.encode(
                batch,
                convert_to_tensor=True,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out_list.append(e.to(dtype=torch.float32).cpu())
    return torch.cat(out_list, dim=0)


def pregenerate_embeddings(
    out_path: Path,
    num_texts: int,
    encode_batch_size: int,
    text_seed: int,
    device: torch.device,
) -> torch.Tensor:
    out_path = Path(out_path)
    print("Loading MiniLM-L6-v2 …")
    st = load_sentence_transformer(MINILM_MODEL, str(device))
    print("Loading text (WikiText-2) …")
    lines = load_text_lines()
    print(f"lines in pool: {len(lines)}")
    print(f"Pregenerating {num_texts} text embeddings …")
    emb = build_text_embedding_matrix(st, lines, num_texts, encode_batch_size, device, text_seed)
    torch.save(
        {
            "embeddings": emb,
            "model": MINILM_MODEL,
            "num_texts": num_texts,
            "text_seed": text_seed,
            "shape": tuple(emb.shape),
        },
        out_path,
    )
    print(f"saved {out_path} shape={tuple(emb.shape)}")
    return emb


def load_embeddings(path: Path) -> torch.Tensor:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run: python training/text_autoencoder.py pregenerate"
        )
    data = torch.load(path, map_location="cpu", weights_only=False)
    e = data["embeddings"]
    if e.dim() != 2 or e.size(-1) != DIM:
        raise ValueError(f"Expected [N, {DIM}], got {tuple(e.shape)}")
    return e


def train_ae(
    emb: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[Encoder384To64, Decoder64To384]:
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    n = emb.size(0)

    enc = Encoder384To64().to(device)
    dec = Decoder64To384().to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr, weight_decay=1e-4)
    emb_dev = emb.to(device)

    enc.train()
    dec.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), generator=g, device=device)
        x = emb_dev[idx]
        opt.zero_grad(set_to_none=True)
        z = enc(x)
        x_hat = dec(z)
        loss = F.l1_loss(x_hat, x)
        loss.backward()
        opt.step()
        if step == 1 or step % 200 == 0:
            print(f"step {step:5d}  mae {loss.item():.6f}")

    return enc, dec


def save_two_models(
    enc: Encoder384To64,
    dec: Decoder64To384,
    enc_path: Path,
    dec_path: Path,
) -> None:
    torch.save(
        {
            "state_dict": enc.state_dict(),
            "dim": DIM,
            "latent": LATENT,
            "minilm_model": MINILM_MODEL,
            "arch": "Encoder384To64",
        },
        enc_path,
    )
    torch.save(
        {
            "state_dict": dec.state_dict(),
            "dim": DIM,
            "latent": LATENT,
            "minilm_model": MINILM_MODEL,
            "arch": "Decoder64To384",
        },
        dec_path,
    )
    print(f"saved {enc_path}")
    print(f"saved {dec_path}")


def main() -> None:
    dev = select_torch_device()
    p = argparse.ArgumentParser(description="384<->64 text AE (MiniLM embeddings)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("pregenerate", help="Encode texts to [N,384] and save")
    p_pre.add_argument("--out", type=Path, default=DEFAULT_EMB_PATH)
    p_pre.add_argument("--num-texts", type=int, default=20_000)
    p_pre.add_argument("--encode-batch-size", type=int, default=64)
    p_pre.add_argument("--text-seed", type=int, default=0)

    p_tr = sub.add_parser("train", help="Train AE from a pregenerated .pt")
    p_tr.add_argument("--embeddings", type=Path, default=DEFAULT_EMB_PATH)
    p_tr.add_argument("--steps", type=int, default=3000)
    p_tr.add_argument("--batch-size", type=int, default=256)
    p_tr.add_argument("--lr", type=float, default=3e-3)
    p_tr.add_argument("--seed", type=int, default=0)
    p_tr.add_argument("--encoder-out", type=Path, default=CHECKPOINTS_DIR / "ae_encoder_384_to_64.pt")
    p_tr.add_argument("--decoder-out", type=Path, default=CHECKPOINTS_DIR / "ae_decoder_64_to_384.pt")

    p_all = sub.add_parser("all", help="Pregenerate then train")
    p_all.add_argument("--out", type=Path, default=DEFAULT_EMB_PATH)
    p_all.add_argument("--num-texts", type=int, default=20_000)
    p_all.add_argument("--encode-batch-size", type=int, default=64)
    p_all.add_argument("--text-seed", type=int, default=0)
    p_all.add_argument("--steps", type=int, default=3000)
    p_all.add_argument("--batch-size", type=int, default=256)
    p_all.add_argument("--lr", type=float, default=3e-3)
    p_all.add_argument("--seed", type=int, default=0)
    p_all.add_argument("--encoder-out", type=Path, default=CHECKPOINTS_DIR / "ae_encoder_384_to_64.pt")
    p_all.add_argument("--decoder-out", type=Path, default=CHECKPOINTS_DIR / "ae_decoder_64_to_384.pt")

    args = p.parse_args()

    if args.cmd == "pregenerate":
        pregenerate_embeddings(
            args.out,
            num_texts=args.num_texts,
            encode_batch_size=args.encode_batch_size,
            text_seed=args.text_seed,
            device=dev,
        )
        return

    if args.cmd == "train":
        emb = load_embeddings(args.embeddings)
        print(f"embeddings shape: {tuple(emb.shape)}")
        enc, dec = train_ae(emb, args.steps, args.batch_size, args.lr, args.seed, dev)
        save_two_models(enc, dec, args.encoder_out, args.decoder_out)
        return

    pregenerate_embeddings(
        args.out,
        num_texts=args.num_texts,
        encode_batch_size=args.encode_batch_size,
        text_seed=args.text_seed,
        device=dev,
    )
    emb = load_embeddings(args.out)
    enc, dec = train_ae(emb, args.steps, args.batch_size, args.lr, args.seed, dev)
    save_two_models(enc, dec, args.encoder_out, args.decoder_out)


if __name__ == "__main__":
    main()
