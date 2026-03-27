"""
Run four Featherless chat agents with distinct personas.

Each turn: encode (prompt, context) with MiniLM-L6-v2 + pair encoder → 64-D latent,
then call the chat API. API key must come from the environment (never commit secrets).

  export FEATHERLESS_API_KEY="your_key"
  python backend/featherless_agents.py

Optional:
  export FEATHERLESS_BASE_URL="https://api.featherless.ai/v1"
  export FEATHERLESS_MODEL="Qwen/Qwen3-8B"   # default if unset
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from openai import OpenAI

from sentence_transformers import SentenceTransformer

from models.device import select_torch_device
from models.sentence_transformer_loader import load_sentence_transformer
from models.paths import CHECKPOINTS_DIR, resolve_repo_path

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ENCODER = CHECKPOINTS_DIR / "encoder_2x384_to_64.pt"


def _load_pair_encoder(path: Path, device: torch.device) -> nn.Module:
    data = torch.load(path, map_location="cpu", weights_only=False)
    dim = int(data.get("dim", 384))
    latent = int(data.get("latent", 64))
    enc_in = 2 * dim

    class PairEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(enc_in, latent)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x.reshape(x.size(0), -1))

    enc = PairEncoder()
    enc.load_state_dict(data["state_dict"])
    enc.to(device)
    enc.eval()
    return enc


@torch.no_grad()
def encode_prompt_context_64(
    st: SentenceTransformer,
    encoder: nn.Module,
    prompt: str,
    context: str,
    device: torch.device,
) -> torch.Tensor:
    """MiniLM embeddings for prompt + context, stacked → linear encoder → L2-normalized 64-D."""
    ep = st.encode(
        prompt,
        convert_to_tensor=True,
        device=str(device),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    ec = st.encode(
        context,
        convert_to_tensor=True,
        device=str(device),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    x = torch.stack([ep, ec], dim=0).unsqueeze(0).to(dtype=torch.float32)
    z = encoder(x)
    return F.normalize(z.squeeze(0).cpu(), dim=0)


# Four distinct “user agents”: different system identity + task (simulated users).
AGENT_PROFILES: list[dict[str, str]] = [
    {
        "id": "analyst",
        "system": "You are a data analyst who answers in short bullet points and cites uncertainty.",
        "user": "In one paragraph, what are the main risks of overfitting in small datasets?",
        "context": "Machine learning evaluation; classroom explanation.",
    },
    {
        "id": "tutor",
        "system": "You are a patient high-school tutor who uses a simple analogy in every answer.",
        "user": "Explain what a vector embedding is, without jargon walls.",
        "context": "Audience: curious teenager; keep under 120 words.",
    },
    {
        "id": "skeptic",
        "system": "You are a skeptical reviewer: flag common misconceptions and ask one clarifying question.",
        "user": "Is it true that neural networks always need huge data?",
        "context": "Focus on when small data can still work.",
    },
    {
        "id": "builder",
        "system": "You are a pragmatic software engineer who ends with a concrete next step.",
        "user": "How should I structure a tiny Python project that might grow?",
        "context": "Solo dev; prefer conventions over tools.",
    },
]


def main() -> None:
    api_key = os.environ.get("FEATHERLESS_API_KEY")
    if not api_key:
        print("Set FEATHERLESS_API_KEY in the environment.", file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
    model = os.environ.get("FEATHERLESS_MODEL", "Qwen/Qwen3-8B")
    enc_path = resolve_repo_path(os.environ.get("ENCODER_PATH", str(DEFAULT_ENCODER)))

    device = select_torch_device()
    print(f"Torch device: {device}  (TORCH_DEVICE env overrides autodetect)")
    print(f"Loading MiniLM + encoder from {enc_path} …")
    st = load_sentence_transformer(MINILM_MODEL, str(device))
    encoder = _load_pair_encoder(enc_path, device)

    client = OpenAI(base_url=base_url, api_key=api_key)

    print(f"Featherless: {base_url}  model={model}\n")

    summary_rows: list[dict] = []
    for prof in AGENT_PROFILES:
        prompt = prof["user"]
        context = prof["context"]
        z64 = encode_prompt_context_64(st, encoder, prompt, context, device)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prof["system"]},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"},
            ],
        )
        text = response.choices[0].message.content or ""

        print("=" * 60)
        print(f"Agent: {prof['id']}")
        print(f"64-D latent (L2 norm={z64.norm().item():.4f}): {z64[:8].tolist()} …")
        print("-" * 60)
        print(text.strip())
        print()

        summary_rows.append(
            {
                "id": prof["id"],
                "latent_dim": 64,
                "latent_first8": z64[:8].tolist(),
            }
        )

    print("Summary JSON (latent preview = first 8 dims):", json.dumps({"model": model, "agents": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
