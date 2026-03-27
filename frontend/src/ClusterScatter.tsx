import { useId } from "react";
import type { ClusterSnapshot2D } from "./api";

const CLUSTER_COLORS = [
  "#5b9cfa",
  "#3ecf8e",
  "#e3b341",
  "#a78bfa",
  "#f472b6",
  "#67e8f9",
];

const MARGIN = { left: 52, right: 22, top: 36, bottom: 44 };

function boundsFor(points: { x: number; y: number }[]) {
  if (!points.length) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const padX = (maxX - minX) * 0.12 || 0.4;
  const padY = (maxY - minY) * 0.12 || 0.4;
  return {
    minX: minX - padX,
    maxX: maxX + padX,
    minY: minY - padY,
    maxY: maxY + padY,
  };
}

type Props = {
  snap: ClusterSnapshot2D;
  width: number;
  height: number;
  title: string;
};

export function ClusterScatter({ snap, width, height, title }: Props) {
  const rid = useId().replace(/:/g, "");
  const { minX, maxX, minY, maxY } = boundsFor([
    ...snap.points,
    ...snap.centroids,
  ]);

  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;

  const sx = (x: number) =>
    MARGIN.left + ((x - minX) / (maxX - minX || 1)) * plotW;
  const sy = (y: number) =>
    MARGIN.top + plotH - ((y - minY) / (maxY - minY || 1)) * plotH;

  const ticksX = 5;
  const ticksY = 5;
  const xTickVals = Array.from({ length: ticksX + 1 }, (_, i) =>
    minX + ((maxX - minX) * i) / ticksX,
  );
  const yTickVals = Array.from({ length: ticksY + 1 }, (_, i) =>
    minY + ((maxY - minY) * i) / ticksY,
  );

  return (
    <svg
      className="cluster-scatter-svg"
      viewBox={`0 0 ${width} ${height}`}
      width="100%"
      style={{ maxWidth: width, height: "auto", display: "block" }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <radialGradient id={`${rid}-panel`} cx="50%" cy="40%" r="75%">
          <stop offset="0%" stopColor="rgba(20, 32, 52, 0.5)" />
          <stop offset="100%" stopColor="rgba(4, 6, 10, 0.95)" />
        </radialGradient>
        <pattern
          id={`${rid}-minor`}
          width="14"
          height="14"
          patternUnits="userSpaceOnUse"
        >
          <path
            d="M 14 0 L 0 0 0 14"
            fill="none"
            stroke="rgba(80, 110, 150, 0.07)"
            strokeWidth="0.8"
          />
        </pattern>
        <pattern
          id={`${rid}-major`}
          width="42"
          height="42"
          patternUnits="userSpaceOnUse"
        >
          <rect width="42" height="42" fill={`url(#${rid}-minor)`} />
          <path
            d="M 42 0 L 0 0 0 42"
            fill="none"
            stroke="rgba(90, 130, 190, 0.12)"
            strokeWidth="1"
          />
        </pattern>
        <linearGradient id={`${rid}-diag`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="rgba(91, 156, 250, 0.03)" />
          <stop offset="100%" stopColor="rgba(167, 139, 250, 0.02)" />
        </linearGradient>
        <filter id={`${rid}-glow`} x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="2.2" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id={`${rid}-glow-strong`} x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="3.5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        {snap.points.map((p, i) => {
          const fill = CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length];
          return (
            <radialGradient
              key={`g-${p.id}`}
              id={`${rid}-pt-${i}`}
              cx="40%"
              cy="40%"
              r="65%"
            >
              <stop offset="0%" stopColor="#ffffff" stopOpacity="0.95" />
              <stop offset="35%" stopColor={fill} stopOpacity="1" />
              <stop offset="100%" stopColor={fill} stopOpacity="0.85" />
            </radialGradient>
          );
        })}
      </defs>

      <rect width={width} height={height} fill={`url(#${rid}-panel)`} />
      <rect
        x={MARGIN.left}
        y={MARGIN.top}
        width={plotW}
        height={plotH}
        fill={`url(#${rid}-major)`}
        opacity={0.85}
      />
      <rect
        x={MARGIN.left}
        y={MARGIN.top}
        width={plotW}
        height={plotH}
        fill={`url(#${rid}-diag)`}
        opacity={0.9}
      />

      <text
        x={MARGIN.left + 4}
        y={22}
        fill="#9aacbf"
        fontSize={11}
        fontWeight={600}
        fontFamily="DM Sans, system-ui, sans-serif"
      >
        {title}
      </text>

      {/* Plot frame */}
      <rect
        x={MARGIN.left}
        y={MARGIN.top}
        width={plotW}
        height={plotH}
        fill="none"
        stroke="rgba(99, 140, 200, 0.35)"
        strokeWidth={1}
      />

      {/* Y ticks + grid lines */}
      {yTickVals.map((yv, i) => {
        const y = sy(yv);
        return (
          <g key={`yt-${i}`}>
            <line
              x1={MARGIN.left}
              y1={y}
              x2={MARGIN.left + plotW}
              y2={y}
              stroke="rgba(99, 140, 200, 0.08)"
              strokeWidth={i === 0 || i === ticksY ? 0 : 1}
            />
            <text
              x={MARGIN.left - 8}
              y={y + 4}
              textAnchor="end"
              fill="#6b7d95"
              fontSize={9}
              fontFamily="JetBrains Mono, monospace"
            >
              {yv.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* X ticks */}
      {xTickVals.map((xv, i) => {
        const x = sx(xv);
        return (
          <g key={`xt-${i}`}>
            <line
              x1={x}
              y1={MARGIN.top}
              x2={x}
              y2={MARGIN.top + plotH}
              stroke="rgba(99, 140, 200, 0.08)"
              strokeWidth={i === 0 || i === ticksX ? 0 : 1}
            />
            <text
              x={x}
              y={MARGIN.top + plotH + 22}
              textAnchor="middle"
              fill="#6b7d95"
              fontSize={9}
              fontFamily="JetBrains Mono, monospace"
            >
              {xv.toFixed(2)}
            </text>
          </g>
        );
      })}

      <text
        x={MARGIN.left + plotW / 2}
        y={height - 6}
        textAnchor="middle"
        fill="#8b9cb8"
        fontSize={10}
        fontWeight={600}
        fontFamily="JetBrains Mono, monospace"
      >
        PC₁ (2D PCA projection)
      </text>
      <text
        x={14}
        y={MARGIN.top + plotH / 2}
        textAnchor="middle"
        fill="#8b9cb8"
        fontSize={10}
        fontWeight={600}
        fontFamily="JetBrains Mono, monospace"
        transform={`rotate(-90, 14, ${MARGIN.top + plotH / 2})`}
      >
        PC₂
      </text>

      {/* Centroids */}
      {snap.centroids.map((c, i) => (
        <g key={`cent-${i}`}>
          <line
            x1={sx(c.x) - 14}
            y1={sy(c.y)}
            x2={sx(c.x) + 14}
            y2={sy(c.y)}
            stroke="rgba(180, 195, 220, 0.55)"
            strokeWidth={1}
          />
          <line
            x1={sx(c.x)}
            y1={sy(c.y) - 14}
            x2={sx(c.x)}
            y2={sy(c.y) + 14}
            stroke="rgba(180, 195, 220, 0.55)"
            strokeWidth={1}
          />
          <circle
            cx={sx(c.x)}
            cy={sy(c.y)}
            r={17}
            fill="none"
            stroke="rgba(139, 156, 184, 0.65)"
            strokeWidth={1.5}
            strokeDasharray="4 3"
          />
          <rect
            x={sx(c.x) + 20}
            y={sy(c.y) - 10}
            width={36}
            height={18}
            rx={4}
            fill="rgba(12, 18, 28, 0.92)"
            stroke="rgba(99, 140, 200, 0.4)"
          />
          <text
            x={sx(c.x) + 38}
            y={sy(c.y) + 3}
            textAnchor="middle"
            fill="#b8c8e0"
            fontSize={10}
            fontFamily="JetBrains Mono, monospace"
          >
            C{i}
          </text>
        </g>
      ))}

      {/* Points */}
      {snap.points.map((p, i) => {
        const anom = p.is_anomaly;
        const cx = sx(p.x);
        const cy = sy(p.y);
        return (
          <g
            key={p.id}
            filter={anom ? `url(#${rid}-glow-strong)` : `url(#${rid}-glow)`}
          >
            {anom && (
              <>
                <circle
                  cx={cx}
                  cy={cy}
                  r={18}
                  fill="none"
                  stroke="#f47068"
                  strokeWidth={1}
                  opacity={0.35}
                  className="anomaly-ring-outer"
                />
                <circle
                  cx={cx}
                  cy={cy}
                  r={14}
                  fill="none"
                  stroke="#ff9088"
                  strokeWidth={1.5}
                  className="anomaly-ring"
                />
              </>
            )}
            <circle
              cx={cx}
              cy={cy}
              r={anom ? 7 : 5.5}
              fill={`url(#${rid}-pt-${i})`}
              stroke="rgba(255,255,255,0.35)"
              strokeWidth={1.2}
            />
            <circle
              cx={cx}
              cy={cy}
              r={anom ? 3.5 : 2.5}
              fill="#ffffff"
              opacity={0.45}
            />
            <title>
              {p.agent_id}: {p.snippet}
            </title>
          </g>
        );
      })}

      {snap.points.length === 0 && (
        <text
          x={width / 2 - 70}
          y={height / 2}
          fill="#5c6d8a"
          fontSize={13}
        >
          No points
        </text>
      )}
    </svg>
  );
}
