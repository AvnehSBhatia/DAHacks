import { useMemo } from "react";
import type { DemoResult, MorphFrame3D } from "./api";

function hashSeed(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function fmt(n: number, d = 3) {
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(d);
}

function VectorHeatStrip({ values }: { values: number[] }) {
  const max = Math.max(...values.map((x) => Math.abs(x)), 1e-9);
  return (
    <div className="pipeline-heat pipeline-heat--compact" role="img" aria-label="z preview">
      {values.map((x, i) => {
        const t = (x / max + 1) / 2;
        const l = 28 + t * 42;
        return (
          <span
            key={i}
            className="pipeline-heat-cell"
            style={{
              background: `linear-gradient(180deg, hsl(175 85% ${l}%) 0%, hsl(200 70% ${l * 0.85}%) 100%)`,
            }}
            title={`[${i}]=${fmt(x, 3)}`}
          />
        );
      })}
    </div>
  );
}

type Props = {
  data: DemoResult;
  frame: number;
  morphFrame: MorphFrame3D | null;
};

export function PipelineStoryboard({ data, frame, morphFrame }: Props) {
  const ctx = (data.context ?? "").trim();
  const promptBytes = new TextEncoder().encode(data.prompt).length;
  const ctxBytes = ctx ? new TextEncoder().encode(ctx).length : 0;
  const fusedBytes = promptBytes + ctxBytes;

  const embedDim = data.latent?.dim ?? morphFrame?.embed_dim ?? 64;
  const hidden384 = 384;

  const zSource = data.latent?.session_z;
  const zPreviewLen = Math.min(embedDim, 24);
  const zPreview = useMemo(() => {
    if (zSource && zSource.length > 0) {
      return zSource.slice(0, zPreviewLen);
    }
    const out: number[] = [];
    const r = mulberry32(hashSeed(data.prompt + "|z"));
    for (let i = 0; i < zPreviewLen; i++) out.push((r() * 2 - 1) * (0.15 + r() * 0.85));
    return out;
  }, [data.prompt, zSource, zPreviewLen]);

  const synth = useMemo(() => {
    const rng = mulberry32(hashSeed(data.prompt + "|" + (ctx || "")));
    const fakeTokPrompt = Math.max(8, Math.round(promptBytes / 3.1 + rng() * 6));
    const fakeTokCtx = ctx ? Math.max(4, Math.round(ctxBytes / 3.4 + rng() * 5)) : 0;
    const fakeTokFused = fakeTokPrompt + fakeTokCtx;
    const funnelStages = [
      {
        label: "UTF-8 bytes (prompt+ctx)",
        value: Math.min(512, fusedBytes) || Math.round(40 + rng() * 30),
      },
      { label: "BPE tokens (est.)", value: Math.min(256, fakeTokFused) },
      { label: "Encoder hidden (d)", value: hidden384 },
      { label: "Session latent z (d)", value: embedDim },
    ];
    const fusionCos = 0.82 + rng() * 0.12;
    const fusionBits = 4.2 + rng() * 1.1;
    const kvCrush = 0.62 + rng() * 0.15;
    const spectralRatio = 9 + rng() * 4;
    const mergeEst = Math.round(18 + rng() * 40);
    return {
      fakeTokPrompt,
      fakeTokCtx,
      funnelStages,
      fusionCos,
      fusionBits,
      kvCrush,
      spectralRatio,
      mergeEst,
    };
  }, [ctx, ctxBytes, embedDim, fusedBytes, hidden384, promptBytes]);

  const pcaRaw = morphFrame?.explained_variance_ratio?.length
    ? morphFrame.explained_variance_ratio
    : null;
  const pcaSynth = useMemo(() => {
    const rng = mulberry32(hashSeed(data.prompt + "|pca|" + frame));
    return [0.52 + rng() * 0.12, 0.22 + rng() * 0.1, 0.08 + rng() * 0.06];
  }, [data.prompt, frame]);
  const pca = pcaRaw ?? pcaSynth;
  const sumPca = pca.reduce((a, b) => a + b, 0) || 1;
  const pcaN = pca.map((x) => x / sumPca);
  const pcaDep = pca.join(",");
  const stretch = useMemo(() => {
    const rng = mulberry32(hashSeed(data.prompt + "|stretch|" + frame));
    const parts = pcaDep.split(",").map(Number).filter(Number.isFinite);
    const sum = parts.reduce((a, b) => a + b, 0) || 1;
    const n = parts.map((x) => x / sum);
    return [
      (2.8 + rng() * 0.4) / (n[0] || 0.5),
      (1.4 + rng() * 0.25) / (n[1] || 0.3),
      1,
    ];
  }, [data.prompt, frame, pcaDep]);

  const stepSynth = useMemo(
    () =>
      data.steps.map((_, i) => {
        const r = mulberry32(hashSeed(`${data.prompt}|step|${i}`));
        return { sketch: 0.55 + r() * 0.35, wstretch: 1.02 + r() * 0.08 };
      }),
    [data.prompt, data.steps],
  );

  const w0 = data.w_frobenius_delta_start;
  const w1 = data.w_frobenius_delta_end;
  const wDelta = w1 - w0;
  const zNorm = Math.sqrt(zPreview.reduce((s, x) => s + x * x, 0));

  const bByte = synth.funnelStages[0]!;
  const bTok = synth.funnelStages[1]!;
  const bZ = synth.funnelStages[3]!;
  const ctxShort = ctx
    ? `${ctx.slice(0, 36)}${ctx.length > 36 ? "…" : ""} (${ctxBytes}B)`
    : "none";

  const encShort = data.latent?.encoder_loaded ? "on" : "id";
  const headShort = data.latent?.response_net_loaded ? "on" : "—";

  return (
    <div className="pipeline-story pipeline-story--compact">
      <p className="pipeline-lede-kicker">Latent pipeline</p>
      <h2 className="pipeline-lede-title">Compression, fusion &amp; geometry</h2>
      <p className="pipeline-lede-blurb">
        Text → bottleneck <code className="pipeline-code">z</code> → fusion →{" "}
        <code className="pipeline-code">W</code> → PCA view. * = simulated.
      </p>

      <div className="pipeline-compact-grid">
        <section className="pipeline-block" aria-label="Input and compression path">
          <h3 className="pipeline-block-h">1 · Input → funnel</h3>
          <p className="pipeline-block-sum">
            <strong>{data.prompt.length}</strong>c / <strong>{promptBytes}</strong>B · ctx{" "}
            <strong>{ctxShort}</strong> · ~<strong>{synth.fakeTokPrompt}</strong>/
            <strong>{synth.fakeTokCtx}</strong> tok · merges~<strong>{synth.mergeEst}</strong>
            <span className="pipeline-asterisk">*</span>
          </p>
          <p className="pipeline-block-sum pipeline-mono">
            {bByte.value}B → {bTok.value} tok → {hidden384}d → <strong>{bZ.value}</strong>d
          </p>
        </section>

        <section className="pipeline-block" aria-label="Encoder and session vector">
          <h3 className="pipeline-block-h">2 · Encoder &amp; z</h3>
          <p className="pipeline-block-sum pipeline-mono">
            enc <strong>{encShort}</strong> · head <strong>{headShort}</strong> · ‖z‖{" "}
            <strong>{fmt(zNorm, 3)}</strong>
          </p>
          <VectorHeatStrip values={zPreview} />
        </section>

        <section className="pipeline-block" aria-label="Fusion memory and projection">
          <h3 className="pipeline-block-h">3 · Fusion · W · PCA (T{frame + 1})</h3>
          <p className="pipeline-block-sum pipeline-mono">
            fus cos <strong>{fmt(synth.fusionCos, 2)}</strong> · H{" "}
            <strong>{fmt(synth.fusionBits, 1)}</strong>b · KV{" "}
            <strong>{fmt(synth.kvCrush, 2)}</strong>
            <span className="pipeline-asterisk">*</span>
            {" · "}
            ‖ΔW‖ <strong>{fmt(w0, 3)}</strong>→<strong>{fmt(w1, 3)}</strong> (Δ
            <strong>{fmt(wDelta, 3)}</strong>) · σ* <strong>{fmt(synth.spectralRatio, 1)}</strong>:1
            <span className="pipeline-asterisk">*</span>
          </p>
          <p className="pipeline-block-sum pipeline-mono">
            var{" "}
            <strong>
              {fmt(pcaN[0] ?? 0, 2)}/{fmt(pcaN[1] ?? 0, 2)}/{fmt(pcaN[2] ?? 0, 2)}
            </strong>{" "}
            · λ <strong>{fmt(stretch[0], 1)}:{fmt(stretch[1], 1)}:1</strong>
            <span className="pipeline-asterisk">*</span> · |V|{" "}
            <strong>{morphFrame?.points.length ?? 0}</strong>
          </p>
        </section>

        <section className="pipeline-block" aria-label="Frobenius steps">
          <h3 className="pipeline-block-h">4 · ‖ΔW‖ by step</h3>
          <div className="pipeline-w-strip pipeline-w-strip--compact" role="img" aria-label="Frobenius steps">
            {(() => {
              const vals = data.steps.map((s) => Math.abs(s.w_frobenius_delta));
              const mx = Math.max(...vals, 1e-9);
              return data.steps.map((st, i) => {
                const h = (vals[i]! / mx) * 100;
                return (
                  <div key={i} className="pipeline-w-cell">
                    <div
                      className="pipeline-w-bar"
                      style={{ height: `${h}%` }}
                      title={`${i + 1}: ${fmt(st.w_frobenius_delta, 4)}`}
                    />
                    <span className="pipeline-w-idx">{i + 1}</span>
                  </div>
                );
              });
            })()}
          </div>
        </section>

        <section className="pipeline-block pipeline-block--table" aria-label="Agent passes">
          <h3 className="pipeline-block-h">5 · Agents</h3>
          <div className="pipeline-table-wrap pipeline-table-wrap--compact">
            <table className="pipeline-table pipeline-table--compact">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Agent</th>
                  <th>r</th>
                  <th>ΔW</th>
                  <th className="pipeline-th-sim">sim*</th>
                </tr>
              </thead>
              <tbody>
                {data.steps.map((st, i) => {
                  const row = stepSynth[i] ?? { sketch: 0.7, wstretch: 1.05 };
                  return (
                    <tr key={i}>
                      <td className="pipeline-mono">{i + 1}</td>
                      <td className="pipeline-td-agent">{st.agent.name}</td>
                      <td className="pipeline-mono">{fmt(st.reward, 1)}</td>
                      <td className="pipeline-mono">{fmt(st.w_frobenius_delta, 3)}</td>
                      <td className="pipeline-mono pipeline-td-sim">
                        {fmt(row.sketch, 2)}/{fmt(row.wstretch, 2)}×
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
