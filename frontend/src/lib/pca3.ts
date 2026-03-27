/**
 * Fit 3 principal components on row-wise data matrix (n × d), return projector.
 * Uses covariance + power iteration (symmetric); stable for d ≤ 256, n moderate.
 */
import { dot, norm } from "./mathVec";

function centerRows(X: number[][]): { centered: number[][]; mean: number[] } {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) mean[j] += X[i][j] ?? 0;
  }
  for (let j = 0; j < d; j++) mean[j] /= Math.max(1, n);
  const centered = X.map((row) => row.map((v, j) => v - mean[j]));
  return { centered, mean };
}

function covSym(centered: number[][]): number[][] {
  const n = centered.length;
  const d = centered[0]?.length ?? 0;
  const C: number[][] = Array.from({ length: d }, () => new Array(d).fill(0));
  const scale = n > 1 ? 1 / (n - 1) : 1;
  for (let i = 0; i < n; i++) {
    const r = centered[i];
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) {
        C[a][b] += (r[a] ?? 0) * (r[b] ?? 0) * scale;
      }
    }
  }
  for (let a = 0; a < d; a++) {
    for (let b = 0; b < a; b++) C[a][b] = C[b][a];
  }
  return C;
}

function matVecSym(C: number[][], v: number[]): number[] {
  const d = C.length;
  const out = new Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    let s = 0;
    for (let j = 0; j < d; j++) s += C[i][j] * v[j];
    out[i] = s;
  }
  return out;
}

function powerEigen(
  C: number[][],
  iterations: number,
): { lambda: number; vec: number[] } {
  const d = C.length;
  let v = new Array(d).fill(0).map(() => Math.random() - 0.5);
  let nv = norm(v);
  v = v.map((x) => x / nv);
  for (let it = 0; it < iterations; it++) {
    const w = matVecSym(C, v);
    nv = norm(w);
    if (nv < 1e-14) break;
    v = w.map((x) => x / nv);
  }
  const Cv = matVecSym(C, v);
  const lambda = dot(v, Cv);
  return { lambda, vec: v };
}

function deflate(C: number[][], lambda: number, vec: number[][]): number[][] {
  const d = C.length;
  const out = C.map((row) => row.slice());
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      out[i][j] -= lambda * (vec[0][i] ?? 0) * (vec[0][j] ?? 0);
    }
  }
  return out;
}

export type PcaProjector3 = {
  mean: number[];
  /** Rows = 3 principal axes (unit-ish) */
  components: number[][];
  explainedRatio: number[];
  project(row: number[]): [number, number, number];
};

export function fitPca3(rows: number[][], iterations = 80): PcaProjector3 {
  if (!rows.length || !rows[0]?.length) {
    return {
      mean: [],
      components: [[], [], []],
      explainedRatio: [0, 0, 0],
      project: () => [0, 0, 0],
    };
  }
  const d = rows[0].length;
  if (rows.length < 2) {
    const mean = rows[0].slice();
    return {
      mean,
      components: [
        Array.from({ length: d }, (_, j) => (j === 0 ? 1 : 0)),
        Array.from({ length: d }, (_, j) => (j === 1 ? 1 : 0)),
        Array.from({ length: d }, (_, j) => (j === 2 ? 1 : 0)),
      ],
      explainedRatio: [1, 0, 0],
      project(row: number[]) {
        const x = row.map((v, j) => v - (mean[j] ?? 0));
        return [x[0] ?? 0, x[1] ?? 0, x[2] ?? 0];
      },
    };
  }
  const { centered, mean } = centerRows(rows);
  let C = covSym(centered);
  const comps: number[][] = [];
  const lambdas: number[] = [];
  for (let k = 0; k < 3; k++) {
    const { lambda, vec } = powerEigen(C, iterations);
    if (lambda < 1e-18) {
      comps.push(new Array(d).fill(0));
      lambdas.push(0);
      continue;
    }
    comps.push(vec);
    lambdas.push(lambda);
    C = deflate(C, lambda, [vec]);
  }
  const trace = lambdas.reduce((a, b) => a + Math.max(0, b), 0) || 1;
  const explainedRatio = lambdas.map((l) => Math.max(0, l) / trace);

  return {
    mean,
    components: comps,
    explainedRatio,
    project(row: number[]) {
      const x = row.map((v, j) => v - (mean[j] ?? 0));
      const coords: [number, number, number] = [0, 0, 0];
      for (let k = 0; k < 3; k++) {
        const pc = comps[k];
        if (!pc.length) continue;
        let s = 0;
        for (let j = 0; j < d; j++) s += x[j] * (pc[j] ?? 0);
        coords[k] = s;
      }
      return coords;
    },
  };
}
