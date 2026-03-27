import { useId } from "react";

const AGENTS = [
  { id: "alpha", label: "α", name: "Agent α" },
  { id: "beta", label: "β", name: "Agent β" },
  { id: "gamma", label: "γ", name: "Agent γ", halluc: true },
];

function quadControl(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  bulge: number,
) {
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const px = -dy / len;
  const py = dx / len;
  return { cx: mx + px * bulge, cy: my + py * bulge };
}

function quadPath(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  bulge: number,
) {
  const { cx, cy } = quadControl(x1, y1, x2, y2, bulge);
  return `M ${x1} ${y1} Q ${cx} ${cy} ${x2} ${y2}`;
}

/** Point on quadratic Bézier P0 → Q → P1 at t ∈ [0,1]. */
function quadPointAt(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  bulge: number,
  t: number,
) {
  const { cx, cy } = quadControl(x1, y1, x2, y2, bulge);
  const o = 1 - t;
  return {
    x: o * o * x1 + 2 * o * t * cx + t * t * x2,
    y: o * o * y1 + 2 * o * t * cy + t * t * y2,
  };
}

/**
 * Hub-and-spoke with curved edges — green = retrieve (hub→agent), blue = write (agent→hub).
 */
export function AgentTopology({
  activeStep,
  pulse,
}: {
  activeStep: number;
  pulse: { receive: boolean; write: boolean } | undefined;
}) {
  const gid = useId().replace(/:/g, "");
  const hub = { x: 200, y: 138 };
  const hubR = 34;
  const orbitR = 108;
  const agentR = 22;
  const n = AGENTS.length;

  const placed = AGENTS.map((a, i) => {
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / n;
    const ax = hub.x + orbitR * Math.cos(angle);
    const ay = hub.y + orbitR * Math.sin(angle);
    return { ...a, ax, ay, angle };
  });

  return (
    <svg
      className="agent-topology-svg"
      viewBox="0 0 400 280"
      width="100%"
      style={{ maxWidth: 400, height: "auto", display: "block" }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <radialGradient id={`${gid}-panel`} cx="50%" cy="45%" r="78%">
          <stop offset="0%" stopColor="rgba(18, 26, 42, 0.92)" />
          <stop offset="55%" stopColor="rgba(8, 11, 18, 0.97)" />
          <stop offset="100%" stopColor="rgba(2, 4, 8, 1)" />
        </radialGradient>
        <radialGradient id={`${gid}-vignette`} cx="50%" cy="48%" r="65%">
          <stop offset="0%" stopColor="rgba(91, 156, 250, 0.07)" />
          <stop offset="100%" stopColor="rgba(0, 0, 0, 0.55)" />
        </radialGradient>
        <pattern
          id={`${gid}-dots`}
          width="18"
          height="18"
          patternUnits="userSpaceOnUse"
        >
          <circle cx="2" cy="2" r="0.9" fill="rgba(99, 140, 200, 0.12)" />
          <circle cx="11" cy="11" r="0.7" fill="rgba(99, 140, 200, 0.08)" />
        </pattern>
        <radialGradient id={`${gid}-hub`} cx="38%" cy="32%" r="70%">
          <stop offset="0%" stopColor="rgba(91, 156, 250, 0.45)" />
          <stop offset="100%" stopColor="rgba(14, 18, 28, 0.98)" />
        </radialGradient>
        <filter id={`${gid}-glow-edge`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="2.5" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id={`${gid}-glow-node`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <marker
          id={`${gid}-arr-g`}
          markerWidth="8"
          markerHeight="8"
          refX="7"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L8,4 L0,8 z" fill="rgba(62, 207, 142, 0.9)" />
        </marker>
        <marker
          id={`${gid}-arr-b`}
          markerWidth="8"
          markerHeight="8"
          refX="7"
          refY="4"
          orient="auto"
        >
          <path d="M0,0 L8,4 L0,8 z" fill="rgba(91, 156, 250, 0.95)" />
        </marker>
      </defs>

      <rect width="400" height="280" fill={`url(#${gid}-panel)`} rx="8" />
      <rect width="400" height="280" fill={`url(#${gid}-dots)`} opacity={0.85} rx="8" />
      <rect width="400" height="280" fill={`url(#${gid}-vignette)`} rx="8" />

      <text
        x={16}
        y={26}
        fill="#8b9cb8"
        fontSize={12}
        fontWeight={500}
        fontFamily="DM Sans, system-ui, sans-serif"
      >
        Data flow (per agent step)
      </text>

      <circle
        cx={hub.x}
        cy={hub.y}
        r={hubR + 5}
        fill="none"
        stroke="rgba(91, 156, 250, 0.14)"
        strokeWidth={1}
      />
      <circle
        cx={hub.x}
        cy={hub.y}
        r={hubR}
        fill={`url(#${gid}-hub)`}
        stroke="rgba(91, 156, 250, 0.5)"
        strokeWidth={1.5}
      />
      <text
        x={hub.x}
        y={hub.y - 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#e8edf5"
        fontSize={10}
        fontWeight={600}
      >
        Shared
      </text>
      <text
        x={hub.x}
        y={hub.y + 9}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#e8edf5"
        fontSize={10}
        fontWeight={600}
      >
        memory
      </text>
      <text
        x={hub.x}
        y={hub.y + 22}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#5c6d8a"
        fontSize={8.5}
        fontFamily="JetBrains Mono, monospace"
      >
        + latent W
      </text>

      <text x={16} y={258} fill="#5c6d8a" fontSize={10}>
        <tspan fill="rgba(62, 207, 142, 0.95)">━━</tspan> retrieve top-k ·{" "}
        <tspan fill="rgba(91, 156, 250, 0.95)">━━</tspan> write experience
      </text>

      {placed.map((a, i) => {
        const isActive = i === activeStep;
        const dx = a.ax - hub.x;
        const dy = a.ay - hub.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len;
        const uy = dy / len;

        const hubEdge = {
          x: hub.x + ux * hubR,
          y: hub.y + uy * hubR,
        };
        const agentEdge = {
          x: a.ax - ux * agentR,
          y: a.ay - uy * agentR,
        };

        const bulgeRecv = 22;
        const bulgeWrite = -22;
        const dRecv = quadPath(
          hubEdge.x,
          hubEdge.y,
          agentEdge.x,
          agentEdge.y,
          bulgeRecv,
        );
        const dWrite = quadPath(
          agentEdge.x,
          agentEdge.y,
          hubEdge.x,
          hubEdge.y,
          bulgeWrite,
        );

        const strokeG = isActive
          ? "rgba(62, 207, 142, 0.92)"
          : "rgba(62, 207, 142, 0.22)";
        const strokeB = isActive
          ? "rgba(91, 156, 250, 0.95)"
          : "rgba(91, 156, 250, 0.22)";
        const sw = isActive ? 2.4 : 1.3;
        const op = isActive ? 1 : 0.38;
        const edgeFilter = isActive ? `url(#${gid}-glow-edge)` : undefined;

        const recv = isActive && pulse?.receive;
        const send = isActive && pulse?.write;

        const recvPt = quadPointAt(
          hubEdge.x,
          hubEdge.y,
          agentEdge.x,
          agentEdge.y,
          bulgeRecv,
          0.58,
        );
        const writePt = quadPointAt(
          agentEdge.x,
          agentEdge.y,
          hubEdge.x,
          hubEdge.y,
          bulgeWrite,
          0.42,
        );

        return (
          <g key={a.id}>
            <path
              d={dRecv}
              fill="none"
              stroke={strokeG}
              strokeWidth={sw}
              opacity={op}
              markerEnd={`url(#${gid}-arr-g)`}
              filter={edgeFilter}
            />
            <path
              d={dWrite}
              fill="none"
              stroke={strokeB}
              strokeWidth={sw}
              opacity={op}
              markerEnd={`url(#${gid}-arr-b)`}
              filter={edgeFilter}
            />
            <g filter={isActive ? `url(#${gid}-glow-node)` : undefined}>
              <circle
                cx={a.ax}
                cy={a.ay}
                r={agentR}
                fill="rgba(12, 16, 24, 0.96)"
                stroke={
                  a.halluc
                    ? "rgba(227, 179, 65, 0.88)"
                    : "rgba(99, 140, 200, 0.4)"
                }
                strokeWidth={isActive ? 2.6 : 2}
              />
            </g>
            {!a.halluc ? (
              <text
                x={a.ax}
                y={a.ay}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#e8edf5"
                fontSize={15}
                fontWeight={700}
              >
                {a.label}
              </text>
            ) : (
              <g>
                <title>Agent γ — hallucination-prone</title>
                <text
                  x={a.ax}
                  y={a.ay - 5}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="#e8edf5"
                  fontSize={14}
                  fontWeight={700}
                >
                  {a.label}
                </text>
                <text
                  x={a.ax}
                  y={a.ay + 8}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="#e3b341"
                  fontSize={6.75}
                  fontWeight={600}
                  letterSpacing="0.02em"
                >
                  at risk
                </text>
              </g>
            )}
            {recv && (
              <circle
                cx={recvPt.x}
                cy={recvPt.y}
                r={4}
                fill="#3ecf8e"
                className="pulse-dot"
              />
            )}
            {send && (
              <circle
                cx={writePt.x}
                cy={writePt.y}
                r={4}
                fill="#5b9cfa"
                className="pulse-dot"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
