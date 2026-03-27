import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { LatentPayload } from "../api";
import {
  SIM_DT,
  buildSimState,
  integrate,
  nearestNeighborCos,
  type SimParticle,
  type SimState,
} from "../lib/latentSim";
import { fitPca3, type PcaProjector3 } from "../lib/pca3";

export type InspectInfo = {
  particle: SimParticle;
  position3d: [number, number, number];
  neighborCos: number;
  neighborId: string;
} | null;

type Props = {
  latent: LatentPayload;
  onInspect: (info: InspectInfo) => void;
  hoverThrottleMs?: number;
};

const PALETTE = [0x4a6fa5, 0x5a8f6f, 0xc49a3c, 0x8b6b9e, 0x6b7c8a, 0x9b6b7a];

/** Anomaly: NVIDIA-green accent is avoided; use clear alert red for demo clarity */
const ANOMALY_COLOR = new THREE.Color(0xff2d3d);

function colorForAgent(agentId: string): THREE.Color {
  let h = 0;
  for (let i = 0; i < agentId.length; i++) h = (h * 31 + agentId.charCodeAt(i)) >>> 0;
  return new THREE.Color(PALETTE[h % PALETTE.length]);
}

export function LatentFieldViz({
  latent,
  onInspect,
  hoverThrottleMs = 48,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const simRef = useRef<SimState | null>(null);
  const pcaRef = useRef<PcaProjector3 | null>(null);
  const onInspectRef = useRef(onInspect);
  onInspectRef.current = onInspect;

  const lastHoverEmitRef = useRef(0);
  const selectedIdxRef = useRef<number | null>(null);

  const pca = useMemo(() => {
    const rows: number[][] = [];
    for (const a of latent.anchors_final) rows.push(a.vector);
    rows.push(latent.ground_truth, latent.base_vector, latent.session_z);
    return fitPca3(rows);
  }, [latent]);

  useEffect(() => {
    pcaRef.current = pca;
  }, [pca]);

  useEffect(() => {
    simRef.current = buildSimState(latent);
    selectedIdxRef.current = null;
  }, [latent]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const emit = (idx: number | null) => {
      const sim = simRef.current;
      const pcaP = pcaRef.current;
      const cb = onInspectRef.current;
      if (!sim || !pcaP || idx === null || idx < 0 || idx >= sim.particles.length) {
        cb(null);
        return;
      }
      const p = sim.particles[idx];
      const pos = pcaP.project(p.v) as [number, number, number];
      const nn = nearestNeighborCos(sim.particles, idx);
      cb({
        particle: p,
        position3d: pos,
        neighborCos: nn.cos,
        neighborId: nn.id,
      });
    };

    const width = el.clientWidth || 640;
    const height = el.clientHeight || 420;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf8fafc);
    scene.fog = new THREE.FogExp2(0xf8fafc, 0.042);

    const camera = new THREE.PerspectiveCamera(48, width / height, 0.08, 120);
    camera.position.set(2.35, 1.55, 2.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.07;
    controls.maxDistance = 22;
    controls.minDistance = 0.35;

    scene.add(new THREE.HemisphereLight(0xfff8ee, 0x7a6a58, 0.95));
    const dir = new THREE.DirectionalLight(0xffffff, 0.55);
    dir.position.set(3.5, 6, 2.5);
    scene.add(dir);
    const fill = new THREE.DirectionalLight(0xe8f0ff, 0.22);
    fill.position.set(-2, 2, -4);
    scene.add(fill);

    const gtGeom = new THREE.SphereGeometry(0.1, 36, 32);
    const gtMat = new THREE.MeshStandardMaterial({
      color: 0xc4a574,
      emissive: 0x5a3d22,
      emissiveIntensity: 0.45,
      metalness: 0.12,
      roughness: 0.38,
    });
    const gtMesh = new THREE.Mesh(gtGeom, gtMat);
    scene.add(gtMesh);

    const baseGeom = new THREE.SphereGeometry(0.062, 22, 18);
    const baseMesh = new THREE.Mesh(
      baseGeom,
      new THREE.MeshStandardMaterial({
        color: 0x8a96a8,
        metalness: 0.1,
        roughness: 0.48,
      }),
    );
    scene.add(baseMesh);

    const sessGeom = new THREE.SphereGeometry(0.052, 18, 16);
    const sessMesh = new THREE.Mesh(
      sessGeom,
      new THREE.MeshStandardMaterial({
        color: 0x5a7a9e,
        metalness: 0.08,
        roughness: 0.5,
      }),
    );
    scene.add(sessMesh);

    const nInst = Math.max(8, latent.anchors_final.length);
    const instGeom = new THREE.SphereGeometry(1, 24, 20);
    const instMat = new THREE.MeshStandardMaterial({
      metalness: 0.1,
      roughness: 0.2,
      transparent: true,
      opacity: 0.95,
      emissive: new THREE.Color(0x000000),
      emissiveIntensity: 0.0,
      vertexColors: true,
    });
    const inst = new THREE.InstancedMesh(instGeom, instMat, nInst);
    inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    inst.count = latent.anchors_final.length;
    for (let i = 0; i < latent.anchors_final.length; i++) {
      inst.setColorAt(i, colorForAgent(latent.anchors_final[i].agent_id));
    }
    if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
    scene.add(inst);

    /* ── Live coordinate scaffolding: grids + axes + bounding box (lerp-stretched) ── */
    const gridXZ = new THREE.GridHelper(1, 14, 0x9aa8b8, 0xd8cfc0);
    (gridXZ.material as THREE.Material).transparent = true;
    (gridXZ.material as THREE.Material).opacity = 0.55;
    const gridXY = new THREE.GridHelper(1, 14, 0xa8b0c0, 0xd0c8bc);
    gridXY.rotation.z = Math.PI / 2;
    (gridXY.material as THREE.Material).transparent = true;
    (gridXY.material as THREE.Material).opacity = 0.38;
    const gridYZ = new THREE.GridHelper(1, 14, 0xa8b0c0, 0xd0c8bc);
    gridYZ.rotation.x = Math.PI / 2;
    (gridYZ.material as THREE.Material).transparent = true;
    (gridYZ.material as THREE.Material).opacity = 0.38;

    const scaffold = new THREE.Group();
    scaffold.add(gridXZ, gridXY, gridYZ);
    scene.add(scaffold);

    const axes = new THREE.AxesHelper(1);
    (axes.material as THREE.Material).transparent = true;
    (axes.material as THREE.Material).opacity = 0.85;
    scene.add(axes);

    const boxGeo = new THREE.BoxGeometry(1, 1, 1);
    const boxEdges = new THREE.EdgesGeometry(boxGeo);
    const bboxWire = new THREE.LineSegments(
      boxEdges,
      new THREE.LineBasicMaterial({
        color: 0x4a6fa5,
        transparent: true,
        opacity: 0.55,
        depthTest: true,
      }),
    );
    scene.add(bboxWire);

    let smCenter = new THREE.Vector3(0, 0, 0);
    let smExtent = 1.2;

    /* Anomaly → GT pull lines (red) */
    const maxSegVerts = Math.max(nInst * 2, 4);
    const anomalyPos = new Float32Array(maxSegVerts * 3);
    const anomalyGeo = new THREE.BufferGeometry();
    anomalyGeo.setAttribute("position", new THREE.BufferAttribute(anomalyPos, 3));
    const anomalyLines = new THREE.LineSegments(
      anomalyGeo,
      new THREE.LineBasicMaterial({
        color: 0xff1a2e,
        transparent: true,
        opacity: 0.72,
        linewidth: 1,
        depthTest: true,
      }),
    );
    scene.add(anomalyLines);

    /* Anomaly rings (torus at anchor) */
    const ringGeom = new THREE.TorusGeometry(1, 0.04, 10, 28);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xff2233,
      transparent: true,
      opacity: 0.65,
      depthTest: true,
    });
    const anomalyRings: THREE.Mesh[] = [];
    const maxRings = Math.min(24, nInst);
    for (let i = 0; i < maxRings; i++) {
      const m = new THREE.Mesh(ringGeom, ringMat.clone());
      m.visible = false;
      m.rotation.x = Math.PI / 2;
      scene.add(m);
      anomalyRings.push(m);
    }

    const dummy = new THREE.Object3D();
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const tmpBox = new THREE.Box3();
    const tmpVec = new THREE.Vector3();

    let raf = 0;
    let last = performance.now();
    let frameId = 0;

    const updateMatrices = (timeMs: number) => {
      const sim = simRef.current;
      const pcaP = pcaRef.current;
      if (!sim || !pcaP) return;

      const gtP = pcaP.project(sim.gt);
      gtMesh.position.set(gtP[0], gtP[1], gtP[2]);
      const pulse = 1 + 0.06 * Math.sin(timeMs * 0.0022);
      gtMesh.scale.setScalar(pulse);

      const baseP = pcaP.project(sim.base);
      baseMesh.position.set(baseP[0], baseP[1], baseP[2]);
      const sP = pcaP.project(sim.session);
      sessMesh.position.set(sP[0], sP[1], sP[2]);

      const n = sim.particles.length;
      inst.count = n;

      let ai = 0;
      let ringIdx = 0;
      tmpBox.makeEmpty();

      for (let i = 0; i < n; i++) {
        const p = sim.particles[i];
        const [x, y, z] = pcaP.project(p.v);
        tmpVec.set(x, y, z);
        tmpBox.expandByPoint(tmpVec);

        const w = Math.min(1.2, p.weight);
        let s =
          0.04 +
          0.13 * w * (p.penalized ? 0.78 : 1);
        if (p.anomaly) {
          s *= 1.18;
        }
        dummy.position.set(x, y, z);
        dummy.scale.setScalar(s);
        dummy.updateMatrix();
        inst.setMatrixAt(i, dummy.matrix);

        if (p.anomaly) {
          const c = ANOMALY_COLOR.clone();
          c.multiplyScalar(0.55 + 0.45 * Math.min(1, w));
          inst.setColorAt(i, c);
          if (ai + 6 <= anomalyPos.length) {
            anomalyPos[ai] = gtP[0];
            anomalyPos[ai + 1] = gtP[1];
            anomalyPos[ai + 2] = gtP[2];
            anomalyPos[ai + 3] = x;
            anomalyPos[ai + 4] = y;
            anomalyPos[ai + 5] = z;
            ai += 6;
          }
          if (ringIdx < anomalyRings.length) {
            const ring = anomalyRings[ringIdx]!;
            ring.visible = true;
            ring.position.set(x, y, z);
            ring.scale.setScalar(s * 2.4 + 0.08);
            ringIdx++;
          }
        } else {
          const dim = 0.44 + 0.56 * Math.min(1, w);
          inst.setColorAt(i, colorForAgent(p.agentId).multiplyScalar(dim));
        }
      }

      for (let r = ringIdx; r < anomalyRings.length; r++) {
        anomalyRings[r]!.visible = false;
      }

      tmpBox.expandByPoint(new THREE.Vector3(gtP[0], gtP[1], gtP[2]));
      tmpBox.expandByPoint(new THREE.Vector3(baseP[0], baseP[1], baseP[2]));
      tmpBox.expandByPoint(new THREE.Vector3(sP[0], sP[1], sP[2]));

      const center = tmpBox.getCenter(tmpVec);
      const size = tmpBox.getSize(new THREE.Vector3());
      const extent = Math.max(size.x, size.y, size.z, 0.35) * 1.38;

      smCenter.lerp(center, 0.14);
      smExtent = THREE.MathUtils.lerp(smExtent, extent, 0.12);

      scaffold.position.copy(smCenter);
      scaffold.scale.setScalar(smExtent);

      axes.position.copy(smCenter);
      axes.scale.setScalar(smExtent * 0.42);

      bboxWire.position.copy(smCenter);
      bboxWire.scale.copy(size.clone().multiplyScalar(1.08).addScalar(0.02));
      if (bboxWire.scale.x < 0.15) bboxWire.scale.setScalar(0.15);

      const posAttr = anomalyGeo.getAttribute("position") as THREE.BufferAttribute;
      posAttr.array = anomalyPos;
      posAttr.needsUpdate = true;
      anomalyGeo.setDrawRange(0, (ai / 3) | 0);

      inst.instanceMatrix.needsUpdate = true;
      if (inst.instanceColor) inst.instanceColor.needsUpdate = true;
    };

    const onPointer = (clientX: number, clientY: number, select: boolean) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(inst, false);
      if (hits.length && hits[0].instanceId !== undefined) {
        const idx = hits[0].instanceId;
        if (select) {
          selectedIdxRef.current = idx;
          emit(idx);
        } else if (selectedIdxRef.current === null) {
          const now = performance.now();
          if (now - lastHoverEmitRef.current > hoverThrottleMs) {
            lastHoverEmitRef.current = now;
            emit(idx);
          }
        }
      } else if (select) {
        selectedIdxRef.current = null;
        emit(null);
      } else if (selectedIdxRef.current === null) {
        const now = performance.now();
        if (now - lastHoverEmitRef.current > hoverThrottleMs + 40) {
          lastHoverEmitRef.current = now;
          emit(null);
        }
      }
    };

    const onMove = (e: PointerEvent) => {
      if (selectedIdxRef.current !== null) return;
      onPointer(e.clientX, e.clientY, false);
    };
    const onClick = (e: MouseEvent) => {
      onPointer(e.clientX, e.clientY, true);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        selectedIdxRef.current = null;
        emit(null);
      }
    };

    window.addEventListener("keydown", onKey);
    renderer.domElement.addEventListener("pointermove", onMove);
    renderer.domElement.addEventListener("click", onClick);

    const loop = (t: number) => {
      const dt = Math.min(0.09, (t - last) / 1000);
      last = t;
      const substeps = Math.min(220, Math.max(1, Math.floor(dt / SIM_DT)));
      if (simRef.current) integrate(simRef.current, substeps);
      updateMatrices(t);
      controls.update();
      renderer.render(scene, camera);
      frameId++;
      if (selectedIdxRef.current !== null && frameId % 2 === 0) {
        emit(selectedIdxRef.current);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    const onResize = () => {
      if (!containerRef.current) return;
      const w = containerRef.current.clientWidth || width;
      const h = containerRef.current.clientHeight || height;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(el);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      window.removeEventListener("keydown", onKey);
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("click", onClick);
      controls.dispose();
      instGeom.dispose();
      instMat.dispose();
      gtGeom.dispose();
      gtMat.dispose();
      baseGeom.dispose();
      baseMesh.material.dispose();
      sessGeom.dispose();
      sessMesh.material.dispose();
      gridXZ.geometry.dispose();
      (gridXZ.material as THREE.Material).dispose();
      gridXY.geometry.dispose();
      (gridXY.material as THREE.Material).dispose();
      gridYZ.geometry.dispose();
      (gridYZ.material as THREE.Material).dispose();
      axes.geometry.dispose();
      (axes.material as THREE.Material).dispose();
      boxGeo.dispose();
      boxEdges.dispose();
      (bboxWire.material as THREE.Material).dispose();
      anomalyGeo.dispose();
      (anomalyLines.material as THREE.Material).dispose();
      ringGeom.dispose();
      for (const m of anomalyRings) {
        m.geometry.dispose();
        (m.material as THREE.Material).dispose();
      }
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, [latent, hoverThrottleMs]);

  return (
    <div className="latent-field" ref={containerRef} role="img" aria-label="Interactive 3D latent space">
      <div className="latent-field__legend">
        <span>
          <i className="latent-dot latent-dot--gt" /> Ground truth
        </span>
        <span>
          <i className="latent-dot latent-dot--base" /> Base
        </span>
        <span>
          <i className="latent-dot latent-dot--sess" /> Session
        </span>
        <span>
          <i className="latent-dot latent-dot--anom" /> Anomaly
        </span>
        <span className="latent-field__hint">Live grid & box · red = anomaly → GT · Drag · Esc</span>
      </div>
    </div>
  );
}
