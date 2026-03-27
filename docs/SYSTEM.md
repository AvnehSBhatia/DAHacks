# DAHacks system overview

This repo implements a **multi-agent session** where each turn is embedded into a shared **64-dimensional latent space**. Agents read and write **anchors** in that space; the system can **retrieve** similar past anchors, score **anomalies**, and **remove** untrusted agents before a final answer.

---

## Data flow (high level)

1. **Text in** — A *prompt* and optional *context* strings describe the session.
2. **MiniLM** — `sentence-transformers/all-MiniLM-L6-v2` maps each short text to a **384-D** vector (semantic embedding).
3. **Optional learned projection** — If checkpoints are present, a **pair encoder** maps `(prompt_vec, context_vec)` (concatenated 768-D) to **64-D** *session latent*; otherwise a fixed orthonormal projection is used so the demo runs without training.
4. **Optional response net** — A small MLP can map `(response_embedding_384, session_latent_64) → refined 64-D` when inserting agent replies as anchors.
5. **LatentSpace** — Stores **anchors** (64-D vectors + metadata: agent id, text snippet, etc.). Supports **deformation** (moving the session vector toward/away from anchors), **ground-truth** biasing, **k-NN retrieval**, and **per-agent anomaly** statistics.
6. **AgentNetwork** — Tracks **trust scores** and which agents are active; orchestration code may **remove** agents flagged as outliers or below a trust threshold.
7. **Output** — Weighted retrieval over anchors picks context for a **final answer** (stub generators in `demo.py`, or real LLM calls in `featherless_agents.py`).

---

## Directory layout

| Path | Role |
|------|------|
| `models/` | Runtime: `LatentSpace`, `Agent` / `AgentNetwork`, device helpers, checkpoint paths, **SentenceTransformer loader** (local cache first). |
| `models/checkpoints/` | `.pt` files: pair AE, text AE, `response_latent_net.pt` (gitignored except `.gitkeep`). |
| `training/` | Scripts to train autoencoders and `ResponseLatentNet` from MiniLM embeddings. |
| `backend/` | `demo.py` (CLI + fake LLM), `cohesive_system.py` (JSON-driven run), `featherless_agents.py` (Featherless API + same latent stack), `system_config.json`. |
| `run_cohesive.sh` | Convenience entrypoint from repo root. |

---

## Device selection

`models/device.py` picks **MPS** on Apple Silicon when available, else CUDA, else CPU. Override with **`TORCH_DEVICE`** (e.g. `mps`, `cpu`).

---

## Hugging Face / MiniLM loading

`models/sentence_transformer_loader.py` loads MiniLM with **`local_files_only=True` first** so tokenizer init does not block on Hub HTTP when the model is already cached. If the snapshot is missing, it **falls back** to a normal online load. For private or rate-limited Hub use, set **`HF_TOKEN`** as usual.

---

## Configuration (`backend/system_config.json`)

Typical fields: encoder/response checkpoint paths (resolved relative to repo root), agent list, generator style (`honest` / `subtle` / `adversarial`), trust and outlier thresholds, step counts. `cohesive_system.py` reads this and runs the loop described in its module docstring.

---

## Mental model

Think of the **64-D vector** as a compressed “where we are in semantic space” for the conversation. Each agent’s messages become **anchors**; retrieval finds **nearby** past content; anomaly detection flags agents whose behavior is **inconsistent** with the group. The ML pieces (autoencoders + response net) are optional refinements on top of the same MiniLM backbone.
