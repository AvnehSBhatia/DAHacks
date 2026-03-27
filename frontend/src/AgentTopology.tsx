import { useId, useEffect, useState } from "react";

const AGENTS = [
  { id: "alpha", label: "α", name: "Agent α" },
  { id: "beta", label: "β", name: "Agent β" },
  { id: "gamma", label: "γ", name: "Agent γ", halluc: true },
];

/**
 * Hub-and-spoke with sharp angular edges — green = retrieve (hub→agent), blue = write (agent→hub).
 */
export function AgentTopology({
  activeStep,
  pulse,
  plotHeight = 320,
}: {
  /** 0–2 highlights α/β/γ; -1 = none (e.g. run has more than three steps). */
  activeStep: number;
  pulse: { receive: boolean; write: boolean } | undefined;
  /** Match VectorSpace3D canvas height in side-by-side layout. */
  plotHeight?: number;
}) {
  const gid = useId().replace(/:/g, "");
  const hub = { x: 200, y: 138 };
  const hubW = 80;
  const hubH = 50;
  const orbitR = 108;
  const agentSize = 40;
  const n = AGENTS.length;

  const [time, setTime] = useState(0);
  useEffect(() => {
    let raf = requestAnimationFrame(function loop() {
      setTime(performance.now() / 1000);
      raf = requestAnimationFrame(loop);
    });
    return () => cancelAnimationFrame(raf);
  }, []);

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
      height={plotHeight}
      style={{
        width: "100%",
        height: plotHeight,
        display: "block",
        maxWidth: "none",
      }}
      preserveAspectRatio="xMidYMid meet"
    >
      <defs>
        <pattern
          id={`${gid}-grid`}
          width="20"
          height="20"
          patternUnits="userSpaceOnUse"
        >
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(0, 255, 204, 0.05)" strokeWidth="1" />
        </pattern>
        <filter id={`${gid}-glow`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id={`${gid}-glow-intense`} x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="6" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <marker
          id={`${gid}-arr-g`}
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="5"
          orient="auto"
        >
          <path d="M0,0 L10,5 L0,10 z" fill="#00ffcc" />
        </marker>
        <marker
          id={`${gid}-arr-b`}
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="5"
          orient="auto"
        >
          <path d="M0,0 L10,5 L0,10 z" fill="#00aaff" />
        </marker>
      </defs>

      <rect width="400" height="280" fill={`url(#${gid}-grid)`} />

      <text
        x={16}
        y={26}
        fill="#00ffcc"
        fontSize={10}
        fontFamily="var(--font-mono)"
        letterSpacing="0.1em"
        opacity={0.7}
      >
        DATA SINK ROUTING
      </text>

      {/* Central Hub Rectangle */}
      <rect
        x={hub.x - hubW / 2}
        y={hub.y - hubH / 2}
        width={hubW}
        height={hubH}
        fill="rgba(0, 255, 204, 0.05)"
        stroke="#00ffcc"
        strokeWidth={1.5}
        filter={`url(#${gid}-glow)`}
      />
      <text
        x={hub.x}
        y={hub.y - 4}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#ffffff"
        fontSize={10}
        fontFamily="var(--font-mono)"
        letterSpacing="0.1em"
      >
        SHARED
      </text>
      <text
        x={hub.x}
        y={hub.y + 8}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#ffffff"
        fontSize={10}
        fontFamily="var(--font-mono)"
        letterSpacing="0.1em"
      >
        MEMORY
      </text>

      <text x={16} y={258} fill="var(--ink-muted)" fontSize={9} fontFamily="var(--font-mono)">
        <tspan fill="#00ffcc">━━</tspan> RETRIEVE <tspan fill="#00aaff" dx="10">━━</tspan> WRITE
      </text>

      {placed.map((a, i) => {
        const isActive = i === activeStep;
        const dx = a.ax - hub.x;
        const dy = a.ay - hub.y;
        const len = Math.hypot(dx, dy) || 1;
        const ux = dx / len;
        const uy = dy / len;

        const hubEdge = {
          x: hub.x + ux * (hubW / 2 + 5),
          y: hub.y + uy * (hubH / 2 + 5),
        };
        const agentEdge = {
          x: a.ax - ux * (agentSize / 2 + 5),
          y: a.ay - uy * (agentSize / 2 + 5),
        };

        const strokeG = isActive ? "#00ffcc" : "rgba(0, 255, 204, 0.2)";
        const strokeB = isActive ? "#00aaff" : "rgba(0, 170, 255, 0.2)";
        const sw = isActive ? 2 : 1;
        const op = isActive ? 1 : 0.4;
        const edgeFilter = isActive ? `url(#${gid}-glow)` : undefined;

        const recv = isActive && pulse?.receive;
        const send = isActive && pulse?.write;

        // Straight lines with angular off-sets instead of curved quad paths
        const ox = -uy * 8; // Perpendicular offset
        const oy = ux * 8;

        const dRecv = `M ${hubEdge.x + ox} ${hubEdge.y + oy} L ${agentEdge.x + ox} ${agentEdge.y + oy}`;
        const dWrite = `M ${agentEdge.x - ox} ${agentEdge.y - oy} L ${hubEdge.x - ox} ${hubEdge.y - oy}`;

        // Pulsing animation variables
        const pulseAnim = Math.abs(Math.sin((time * 4) + i));
        
        // Calculate point along the line strictly for the pulse dot
        const tRecv = (time * 2) % 1;
        const tWrite = (time * 2 + 0.5) % 1;
        const recvPtX = hubEdge.x + ox + (agentEdge.x - hubEdge.x) * tRecv;
        const recvPtY = hubEdge.y + oy + (agentEdge.y - hubEdge.y) * tRecv;
        const writePtX = agentEdge.x - ox + (hubEdge.x - agentEdge.x) * tWrite;
        const writePtY = agentEdge.y - oy + (hubEdge.y - agentEdge.y) * tWrite;

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
              strokeDasharray={isActive ? "4 4" : "none"}
            />
            <path
              d={dWrite}
              fill="none"
              stroke={strokeB}
              strokeWidth={sw}
              opacity={op}
              markerEnd={`url(#${gid}-arr-b)`}
              filter={edgeFilter}
              strokeDasharray={isActive ? "4 4" : "none"}
            />
            
            <g filter={isActive ? `url(#${gid}-glow-intense)` : undefined}>
               {/* Use Diamond/Polygons instead of circles */}
               <polygon 
                 points={`${a.ax},${a.ay - agentSize/2} ${a.ax + agentSize/2},${a.ay} ${a.ax},${a.ay + agentSize/2} ${a.ax - agentSize/2},${a.ay}`}
                 fill="rgba(10, 10, 12, 0.95)"
                 stroke={
                   a.halluc 
                     ? (isActive ? "#ff0033" : "rgba(255, 0, 51, 0.4)")
                     : (isActive ? "#00ffcc" : "rgba(0, 255, 204, 0.4)")
                 }
                 strokeWidth={isActive ? 2 + pulseAnim * 2 : 1}
               />
               {isActive && (
                 <polygon 
                   points={`${a.ax},${a.ay - agentSize/2 - 4} ${a.ax + agentSize/2 + 4},${a.ay} ${a.ax},${a.ay + agentSize/2 + 4} ${a.ax - agentSize/2 - 4},${a.ay}`}
                   fill="none"
                   stroke={a.halluc ? "#ff0033" : "#00ffcc"}
                   strokeWidth={1}
                   opacity={pulseAnim * 0.5}
                 />
               )}
            </g>

            <text
              x={a.ax}
              y={a.ay}
              textAnchor="middle"
              dominantBaseline="middle"
              fill={a.halluc && isActive ? "#ff3366" : "#ffffff"}
              fontSize={14}
              fontFamily="var(--font-mono)"
              fontWeight={700}
            >
              {a.label}
            </text>

            {a.halluc && (
               <text
                 x={a.ax}
                 y={a.ay + agentSize / 2 + 10}
                 textAnchor="middle"
                 dominantBaseline="middle"
                 fill="#ff3366"
                 fontSize={8}
                 fontFamily="var(--font-mono)"
                 letterSpacing="0.1em"
               >
                 [AT_RISK]
               </text>
            )}

            {recv && (
              <rect
                x={recvPtX - 3}
                y={recvPtY - 3}
                width={6}
                height={6}
                fill="#00ffcc"
                filter={`url(#${gid}-glow)`}
              />
            )}
            {send && (
              <rect
                x={writePtX - 3}
                y={writePtY - 3}
                width={6}
                height={6}
                fill="#00aaff"
                filter={`url(#${gid}-glow)`}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
