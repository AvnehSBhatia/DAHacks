type Props = {
  values: number[];
  width?: number;
  height?: number;
};

/** Minimal inline SVG sparkline for weight history */
export function MiniSparkline({ values, width = 200, height = 44 }: Props) {
  if (values.length < 2) {
    return <div className="spark spark--empty">No samples yet</div>;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = 4;
  const w = width - pad * 2;
  const h = height - pad * 2;
  const span = max - min || 1;
  const pts = values.map((v, i) => {
    const x = pad + (i / (values.length - 1)) * w;
    const y = pad + h - ((v - min) / span) * h;
    return `${x},${y}`;
  });
  const d = `M ${pts.join(" L ")}`;

  return (
    <svg className="spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden>
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="var(--spark-from)" />
          <stop offset="100%" stopColor="var(--spark-to)" />
        </linearGradient>
      </defs>
      <path d={d} fill="none" stroke="url(#sparkGrad)" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}
