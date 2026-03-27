/** Small numeric helpers for 64-D latent vectors (no external deps). */

export function dot(a: number[], b: number[]): number {
  let s = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) s += a[i] * b[i];
  return s;
}

export function norm(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

export function normalize(a: number[]): number[] {
  const n = norm(a);
  if (n < 1e-12) return a.slice();
  return a.map((x) => x / n);
}

export function addScaled(a: number[], b: number[], s: number): number[] {
  return a.map((x, i) => x + s * (b[i] ?? 0));
}

export function sub(a: number[], b: number[]): number[] {
  return a.map((x, i) => x - (b[i] ?? 0));
}

export function scale(a: number[], s: number): number[] {
  return a.map((x) => x * s);
}

export function cosineSimilarity(a: number[], b: number[]): number {
  const na = norm(a);
  const nb = norm(b);
  if (na < 1e-12 || nb < 1e-12) return 0;
  return dot(a, b) / (na * nb);
}
