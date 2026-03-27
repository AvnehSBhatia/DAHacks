"""
Map (384D response, 64D latent) -> new 64D vector.

The 64D latent defines a 64-dimensional subspace of R^384 via learned orthonormal
columns W(z) ∈ R^{384×64} (QR factorization). The projection of the response onto
that subspace has coordinates proj = W^T r ∈ R^64.

Loss:
  - Match the network output to that projection (MSE).
  - Align the output with the input latent (1 - cosine similarity).

Run:  python training/response_latent.py --steps 1000
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.device import select_torch_device
from models.paths import CHECKPOINTS_DIR, ensure_checkpoints_dir

ensure_checkpoints_dir()

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 384
LATENT = 64

# Shared with SIGINT handler — set before training starts
_checkpoint_ctx: dict[str, Any] = {}


def batched_orthonormal_basis(z: torch.Tensor, W_head: nn.Sequential) -> torch.Tensor:
    """z [B,64] -> W [B,384,64] with orthonormal columns (batched QR)."""
    b = z.size(0)
    raw = W_head(z).view(b, DIM, LATENT)
    w, _ = torch.linalg.qr(raw, mode="reduced")
    return w


def response_projection_coords(r: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """r [B,384], W [B,384,64] -> W^T r as [B,64]."""
    return torch.einsum("bd,bdk->bk", r, W)


class ResponseLatentNet(nn.Module):
    """
    Inputs: r [B,384] (response embedding), z [B,64] (compressed / latent).
    Output: z_out [B,64].

    Internally builds W(z) with orthonormal columns; fusion MLP maps (r,z) -> z_out.
    """

    def __init__(self, d: int = DIM, latent: int = LATENT, hidden: int = 512) -> None:
        super().__init__()
        self.d = d
        self.latent = latent
        self.W_head = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, d * latent),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d + latent, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, latent),
        )

    def forward(self, r: torch.Tensor, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W = batched_orthonormal_basis(z, self.W_head)
        proj = response_projection_coords(r, W)
        z_out = self.fusion(torch.cat([r, z], dim=-1))
        return z_out, proj, W


def combined_loss(
    z_out: torch.Tensor,
    proj: torch.Tensor,
    z_latent: torch.Tensor,
    lambda_cos: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    proj loss: output should equal projection of r onto span(W(z)).
    cosine loss: output should align with input latent z.
    """
    loss_proj = F.mse_loss(z_out, proj)
    cos = F.cosine_similarity(z_out, z_latent, dim=-1, eps=1e-8)
    loss_cos = (1.0 - cos).mean()
    total = loss_proj + lambda_cos * loss_cos
    return total, loss_proj, loss_cos


def make_pool(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed synthetic pool on CPU: unit-norm r [N,384], z [N,64]."""
    g = torch.Generator().manual_seed(seed)
    r = F.normalize(torch.randn(n, DIM, generator=g), dim=-1, eps=1e-8)
    z = F.normalize(torch.randn(n, LATENT, generator=g), dim=-1, eps=1e-8)
    return r, z


@torch.no_grad()
def evaluate(
    model: ResponseLatentNet,
    r: torch.Tensor,
    z: torch.Tensor,
    device: torch.device,
    batch_size: int,
    lambda_cos: float,
) -> tuple[float, float, float, float]:
    """Returns mean total, mse_proj, cos_term, mean_cos(z_out,z) over full r,z (CPU tensors)."""
    model.eval()
    n = r.size(0)
    tot_acc = 0.0
    lp_acc = 0.0
    lc_acc = 0.0
    cos_acc = 0.0
    seen = 0
    for i in range(0, n, batch_size):
        rb = r[i : i + batch_size].to(device)
        zb = z[i : i + batch_size].to(device)
        z_out, proj, _ = model(rb, zb)
        loss, lp, lc = combined_loss(z_out, proj, zb, lambda_cos)
        cos_m = F.cosine_similarity(z_out, zb, dim=-1).mean()
        bs = rb.size(0)
        tot_acc += loss.item() * bs
        lp_acc += lp.item() * bs
        lc_acc += lc.item() * bs
        cos_acc += cos_m.item() * bs
        seen += bs
    return tot_acc / seen, lp_acc / seen, lc_acc / seen, cos_acc / seen


def save_model(path: str, model: ResponseLatentNet, *, lambda_cos: float, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "dim": DIM,
        "latent": LATENT,
        "arch": "ResponseLatentNet",
        "lambda_cos": lambda_cos,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def _sigint_handler(signum: int, frame: Any) -> None:
    m = _checkpoint_ctx.get("model")
    path = _checkpoint_ctx.get("out")
    lam = _checkpoint_ctx.get("lambda_cos", 0.5)
    step = _checkpoint_ctx.get("step", 0)
    if m is not None and isinstance(path, str):
        save_model(
            path,
            m,
            lambda_cos=float(lam),
            extra={"saved_reason": "sigint", "step": step},
        )
        print(f"\nSaved checkpoint to {path} (Ctrl+C)", file=sys.stderr)
    sys.exit(128 + signum)


def train_loop(
    steps: int,
    batch_size: int,
    lr: float,
    lambda_cos: float,
    pool_size: int,
    val_fraction: float,
    pool_seed: int,
    val_every: int,
    val_batch_size: int | None,
    device: str | None,
    seed: int,
    out_path: str,
) -> ResponseLatentNet:
    torch.manual_seed(seed)
    dev = select_torch_device(device)
    dtype = torch.float32

    r_all, z_all = make_pool(pool_size, pool_seed)
    n_val = max(1, int(round(pool_size * val_fraction)))
    n_train = pool_size - n_val
    r_train, z_train = r_all[:n_train], z_all[:n_train]
    r_val, z_val = r_all[n_train:], z_all[n_train:]
    print(f"pool={pool_size}  train={n_train}  val={n_val}  val_fraction={val_fraction:.3f}")

    model = ResponseLatentNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    _checkpoint_ctx.clear()
    _checkpoint_ctx.update(
        {
            "model": model,
            "out": out_path,
            "lambda_cos": lambda_cos,
            "step": 0,
        }
    )
    signal.signal(signal.SIGINT, _sigint_handler)

    g_batch = torch.Generator()
    g_batch.manual_seed(seed + 999)

    model.train()
    for step in range(1, steps + 1):
        _checkpoint_ctx["step"] = step
        idx = torch.randint(0, n_train, (batch_size,), generator=g_batch)
        r = r_train[idx].to(dev, dtype=dtype)
        z = z_train[idx].to(dev, dtype=dtype)

        opt.zero_grad(set_to_none=True)
        z_out, proj, _ = model(r, z)
        loss, lp, lc = combined_loss(z_out, proj, z, lambda_cos)
        loss.backward()
        opt.step()

        if step == 1 or step % 200 == 0:
            with torch.no_grad():
                cos_out = F.cosine_similarity(z_out, z, dim=-1).mean()
                print(
                    f"step {step:5d}  train_loss {loss.item():.6f}  "
                    f"mse_proj {lp.item():.6f}  cos_term {lc.item():.6f}  "
                    f"mean_cos(z_out,z) {cos_out.item():.4f}"
                )

        if step % val_every == 0:
            vb = val_batch_size or min(batch_size, n_val)
            vt, vlp, vlc, vcos = evaluate(model, r_val, z_val, dev, vb, lambda_cos)
            print(
                f"          val_loss {vt:.6f}  val_mse_proj {vlp:.6f}  val_cos_term {vlc:.6f}  "
                f"val_mean_cos {vcos:.4f}"
            )
            model.train()

    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Train response+latent -> 64D fusion model")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-cos", type=float, default=0.5, help="Weight for (1 - cos(z_out, z))")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pool-size", type=int, default=50_000, help="Total synthetic (r,z) pairs before split")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of pool for validation")
    p.add_argument("--pool-seed", type=int, default=0, help="RNG seed for generating the fixed pool")
    p.add_argument("--val-every", type=int, default=200, help="Run validation every N steps")
    p.add_argument("--val-batch-size", type=int, default=None, help="Val batch size (default: min(batch, val size))")
    p.add_argument("--out", type=str, default=str(CHECKPOINTS_DIR / "response_latent_net.pt"))
    args = p.parse_args()

    try:
        m = train_loop(
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_cos=args.lambda_cos,
            pool_size=args.pool_size,
            val_fraction=args.val_fraction,
            pool_seed=args.pool_seed,
            val_every=args.val_every,
            val_batch_size=args.val_batch_size,
            device=None,
            seed=args.seed,
            out_path=args.out,
        )
    except KeyboardInterrupt:
        # Fallback if SIGINT was not delivered to our handler (e.g. some environments)
        m = _checkpoint_ctx.get("model")
        if m is not None:
            save_model(args.out, m, lambda_cos=args.lambda_cos, extra={"saved_reason": "keyboardinterrupt"})
            print(f"\nSaved checkpoint to {args.out} (KeyboardInterrupt)", file=sys.stderr)
        raise

    save_model(
        args.out,
        m,
        lambda_cos=args.lambda_cos,
        extra={"saved_reason": "finished", "step": args.steps},
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
