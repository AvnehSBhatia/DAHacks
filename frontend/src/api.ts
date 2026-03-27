export type Point3D = {
  id: string;
  x: number;
  y: number;
  z: number;
  agent_id: string;
  snippet: string;
};

/** 64D vectors → PCA into 3D (morph view — not K-means) */
export type MorphFrame3D = {
  points: Point3D[];
  embed_dim: number;
  explained_variance_ratio: number[];
};

export type SpacePoint2D = {
  id: string;
  x: number;
  y: number;
  cluster: number;
  is_anomaly: boolean;
  agent_id: string;
  snippet: string;
};

export type ClusterSnapshot2D = {
  points: SpacePoint2D[];
  centroids: { x: number; y: number }[];
  anomaly_ids: string[];
  explained_variance_ratio: number[];
};

export type DemoStep = {
  agent: { id: string; name: string; hallucination_prone: boolean };
  action: string;
  reward: number;
  retrieved_snippets: string[];
  pulse: { receive: boolean; write: boolean };
  w_frobenius_delta: number;
};

/** One snapshot of every anchor’s 64-D latent (deformed + original) for custom visuals */
export type AnchorVectorRecord = {
  id: string;
  agent_id: string;
  vector: number[];
  vector_original: number[];
  text: string;
  anomaly: boolean;
  penalized: boolean;
  weight: number;
  impact: number;
  /** Unix seconds from server anchor insertion */
  timestamp?: number;
};

export type LatentPayload = {
  dim: number;
  session_z: number[];
  ground_truth: number[];
  base_vector: number[];
  anchors_final: AnchorVectorRecord[];
  /** Per interaction step: all anchors present after that step */
  timeline_vectors: AnchorVectorRecord[][];
  encoder_loaded: boolean;
  response_net_loaded: boolean;
  num_agents?: number;
  stagger_s?: number;
  cycles?: number;
};

export type DemoResult = {
  prompt: string;
  context?: string;
  steps: DemoStep[];
  morph_frames: MorphFrame3D[];
  final_clusters: ClusterSnapshot2D;
  w_frobenius_delta_start: number;
  w_frobenius_delta_end: number;
  latent?: LatentPayload;
};

export type RunDemoBody = {
  prompt: string;
  context?: string;
  num_agents?: number;
  stagger_s?: number;
  cycles?: number;
};

export async function runDemo(
  prompt: string,
  getAccessToken?: () => Promise<string>,
  context: string = "",
  opts?: Pick<RunDemoBody, "num_agents" | "stagger_s" | "cycles">,
): Promise<DemoResult> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (getAccessToken) {
    const token = await getAccessToken();
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch("/api/demo/run", {
    method: "POST",
    headers,
    body: JSON.stringify({
      prompt,
      context,
      num_agents: opts?.num_agents ?? 3,
      stagger_s: opts?.stagger_s ?? 0.5,
      cycles: opts?.cycles ?? 3,
    }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}

/** 
 * Generator to stream Server-Sent Events (NDJSON) updates from the backend 
 */
export async function* runDemoStream(
  prompt: string,
  context: string = "",
  opts?: Pick<RunDemoBody, "num_agents" | "stagger_s" | "cycles">,
  getAccessToken?: () => Promise<string>,
): AsyncGenerator<any, void, unknown> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (getAccessToken) {
    const token = await getAccessToken();
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch("/api/demo/stream", {
    method: "POST",
    headers,
    body: JSON.stringify({
      prompt,
      context,
      num_agents: opts?.num_agents ?? 3,
      stagger_s: opts?.stagger_s ?? 0.5,
      cycles: opts?.cycles ?? 3,
    }),
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }

  if (!res.body) throw new Error("No response body");
  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    
    // Process SSE properly formatted with `data: ...\n\n`
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() || "";
    
    for (const chunk of chunks) {
      if (chunk.startsWith("data: ")) {
        const jsonStr = chunk.substring("data: ".length).trim();
        if (jsonStr) {
          try {
            yield JSON.parse(jsonStr);
          } catch (e) {
            console.error("Failed to parse chunk:", jsonStr, e);
          }
        }
      }
    }
  }
}
