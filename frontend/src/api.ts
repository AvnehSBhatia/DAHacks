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

export type DemoResult = {
  prompt: string;
  steps: DemoStep[];
  morph_frames: MorphFrame3D[];
  final_clusters: ClusterSnapshot2D;
  w_frobenius_delta_start: number;
  w_frobenius_delta_end: number;
};

export async function runDemo(prompt: string): Promise<DemoResult> {
  const res = await fetch("/api/demo/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(t || res.statusText);
  }
  return res.json();
}
