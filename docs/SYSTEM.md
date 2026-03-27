# DAHacks system overview

This repo implements a **multi-agent session** where each turn is embedded into a shared **64-dimensional latent space**. Agents read and write **anchors** in that space; the system can **retrieve** similar past anchors, score **anomalies**, assign **per-agent health** and **trust**, **penalize** weak agents’ anchors, and **unregister** agents before a final answer.

---

## End-to-end pipeline (cohesive run)

### Phase 0 — Setup

1. **Load models** — `SentenceTransformer` (MiniLM 384-D), optional **pair encoder** `(prompt_emb, context_emb) → 64-D`, optional **ResponseLatentNet** `(response_384, latent_64) → refined 64-D` for inserts.
2. **Base vector** — A short system string is embedded and projected to 64-D; this anchors the **ground-truth (GT)** direction used for deformation and gravity.
3. **Register agents** — Each agent has a **role** string → embedded → projected to a fixed **query vector** `qv` used for retrieval before that agent speaks.
4. **Session vector** — The user **prompt** and **context** are encoded with the same pair path as training: `z_session = embed_pair_to_latent(prompt, context)`. This is the geometric “session intent” for the rest of the run.

### Phase 1 — Pull / push (interaction cycles)

For each cycle and each agent (in order):

1. **Retrieve** — `retrieve(qv, k)` ranks anchors by **cosine similarity × time-decayed weight × impact**, with an extra **0.5×** if the anchor is **post-hoc penalized** (see Phase 2).
2. **Generate** — `generate_fn(prompt, retrieved_texts)` produces a reply (stub or real LLM).
3. **Embed insert** — The reply is turned into a 64-D vector (response net path combines prompt + context + output, or `embed_pair_to_latent(prompt, output)` without a response net).
4. **Insert** — `check_anomaly(z)` runs; if the space has fewer than `min_anchors_for_detection` anchors, anomaly is **skipped** (warm-up). Otherwise a **composite** score compares `z` to k-nearest anchor directions and to **GT**:
   - `composite = 0.6 × (1 − mean_k cos(z, anchor)) + 0.4 × (1 − cos(z, GT))`
   - Above `anomaly_threshold` → anchor marked **anomaly**, downweighted, compressed along the consensus axis at insert.
5. **Periodic** — Every `update_every` steps, `update_cycle()` recomputes GT, applies **gravitational pull** toward GT on original vectors, **decays** weights and stretch, and normalizes.
6. **Trust EMA** — After updates, each agent’s **trust_score** is smoothed toward **`health`** (same formula as below).

### Phase 2 — Prune, penalize, remove

For each **removal pass** (up to `removal_passes`), an agent is **unregistered** if any of these hold:

| Rule | Meaning |
|------|---------|
| **bad_actor** | `agent_anomaly_score` verdict from per-anchor rates vs GT (see below). |
| **Low trust** | Smoothed `trust_score < min_trust`. |
| **Low health** | `health ≤ prune_health_at_or_below` (default **0.5**): composite agent quality in the bottom half of the 0–1 scale. |
| **IQR outlier** | Among ≥3 agents, anomaly_rate above `Q3 + iqr_factor × IQR`. |

**Penalize before unregister:** If the agent is removed and **either** `verdict == bad_actor` **or** `health ≤ prune_health_at_or_below`, all anchors from that agent that are not yet `penalized` get `impact` and `weight` multiplied by **0.5** and `penalized=True`. Those anchors **retain** in the store (history) but contribute **half** as much to retrieval scores and **slow** gravity updates (GT still uses penalized anchors with reduced weight in `agent_anomaly_score` via `penalty_factor` / deformation).

### Phase 3 — Final answer

A **random surviving** agent is chosen. **Weighted retrieval** uses **`z_session`** (not the agent’s role vector) to rank anchors: same score as in Phase 1. Output is either **formatted ranked anchors** (`synthesize=retrieval`) or **`generate_fn(prompt, anchor_texts)`** (`synthesize=agent`).

---

## Per-agent metrics (latent space)

For each agent, consider only anchors tagged with that `agent_id`:

- **`anomaly_rate`** — Fraction of anchors that failed the insert-time anomaly gate.
- **`gt_divergence`** — `1 − cos(centroid(agent anchors), GT)` — how far the agent’s cluster is from global consensus.
- **`health`** — `1 − (0.6 × anomaly_rate + 0.4 × gt_divergence)` ∈ [0, 1]. **Higher is better.**  
  When **`health ≤ 0.5`**, the agent is at least **suspicious** (unless already **bad_actor**).
- **`verdict`** — **bad_actor** if `anomaly_rate ≥ 0.5` or `gt_divergence > 0.6`; else **suspicious** if `health ≤ 0.5` or milder thresholds (`anomaly_rate > 0.25` or `gt_div > 0.35`); else **clean**.

**Trust** (`AgentNetwork`) is an EMA toward `health`:  
`trust ← 0.85 × trust + 0.15 × health`.

---

## Ground truth and deformation

- **GT** is a weighted sum of anchor vectors (anomalies and penalized anchors contribute **less**), normalized.
- **Consensus axis** `n̂ ∝ GT − base_vector` drives **stretch** on insert: “clean” anchors elongate toward consensus; anomalous ones compress.
- **Gravity** pulls each anchor’s **pre-deformation** vector toward GT over time; deformation is reapplied with decaying stretch.

---

## Data flow (embedding)

1. **Text in** — Prompt and context strings.
2. **MiniLM** — 384-D embeddings.
3. **Pair encoder** (optional) — 768-D concat → 64-D session / pair latent.
4. **Response net** (optional) — Fuses 384-D reply embedding with 64-D context latent for insert.
5. **Fallback** — Orthonormal random projection 384→64 when no encoder is loaded.

---

## Directory layout

| Path | Role |
|------|------|
| `models/` | `LatentSpace`, `Agent` / `AgentNetwork`, device helpers, checkpoint paths, **SentenceTransformer loader** (local cache first). |
| `models/checkpoints/` | `.pt` files: pair AE, `response_latent_net.pt` (gitignored except `.gitkeep`). |
| `training/` | Train autoencoders and `ResponseLatentNet`. |
| `backend/` | `demo.py` (generators), `cohesive_system.py`, `featherless_agents.py`, `system_config.json`. |
| `run_cohesive.sh` | Convenience entrypoint from repo root. |

---

## Device selection

`models/device.py` picks **MPS** on Apple Silicon when available, else CUDA, else CPU. Override with **`TORCH_DEVICE`**.

---

## Hugging Face / MiniLM

`models/sentence_transformer_loader.py` tries **`local_files_only=True`** first, then falls back to Hub download. Set **`HF_TOKEN`** if needed.

---

## Configuration (`backend/system_config.json`)

- **paths** — Encoder and response-net checkpoints.
- **session** — Prompt, context, cycles, `final_synthesize`, `retrieval_k`, etc.
- **outliers** — `min_trust`, `iqr_factor`, `removal_passes`, **`prune_health_at_or_below`** (default **0.5**): unregister agents whose **health** is in the bottom band (and penalize their anchors when rules above apply).

---

## Web demo (FastAPI + Vite)

- **API** — `demo/server.py`: `POST /api/demo/run` with JSON `{ "prompt", "context", "num_agents" (default 10), "stagger_s" (default 0.5), "cycles" (default 1) }` runs `demo/latent_demo.py` (real `LatentSpace` + N agents). Between each agent step the server sleeps `stagger_s` seconds (except before the first). Response includes PCA-friendly `morph_frames` / `final_clusters` and **`latent`** with full **64-D** arrays plus `timeline_vectors`.
- **Port** — Uvicorn on **`127.0.0.1:5005`** (see `demo/run_server.sh`). The Vite dev server proxies `/api` and `/health` to that port (`frontend/vite.config.ts`).
- **Run** — Terminal 1: `bash demo/run_server.sh` (or `python -m uvicorn demo.server:app --port 5005`). Terminal 2: `cd frontend && npm run dev`.
- **UI** — Light *old lace* shell (DM Sans): WebGL field with PCA₃ projection, client-side gravity/decay at `Δt=0.001` (batched per animation frame), GT/base/session markers, hover/click detail panel with weight sparkline, and a dark inset for the 2D cluster chart.

---

## Mental model

The **64-D vector** is a compressed semantic location for the session and each reply. **Anchors** are memories; **retrieval** is similarity search with decay and impact; **anomaly** and **health** encode **consensus** vs **outlier** behavior; **prune at health ≤ 0.5** implements “bottom half of the health scale” as **suspicious** and **removal-eligible**, with **anchor penalties** so bad history does not dominate retrieval after the agent is gone.
