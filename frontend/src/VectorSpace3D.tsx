import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { ConvexGeometry } from "three/examples/jsm/geometries/ConvexGeometry.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import type { MorphFrame3D } from "./api";

function normalize(
  points: MorphFrame3D["points"],
): { x: number; y: number; z: number }[] {
  if (points.length === 0) return [];
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity,
    minZ = Infinity,
    maxZ = -Infinity;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
    minZ = Math.min(minZ, p.z);
    maxZ = Math.max(maxZ, p.z);
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const cz = (minZ + maxZ) / 2;
  const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1e-12);
  const scale = 1.65 / span;
  return points.map((p) => ({
    x: (p.x - cx) * scale,
    y: (p.y - cy) * scale,
    z: (p.z - cz) * scale,
  }));
}

function dedupeVectors(vectors: THREE.Vector3[]): THREE.Vector3[] {
  const seen = new Set<string>();
  const out: THREE.Vector3[] = [];
  for (const v of vectors) {
    const k = `${v.x.toFixed(7)},${v.y.toFixed(7)},${v.z.toFixed(7)}`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(v.clone());
  }
  return out;
}

function buildConvexSolidGeometry(
  unique: THREE.Vector3[],
): THREE.BufferGeometry | null {
  const n = unique.length;
  if (n === 0) return null;
  if (n === 1) {
    return new THREE.SphereGeometry(0.14, 32, 32);
  }
  if (n === 2) {
    const a = unique[0];
    const b = unique[1];
    const len = Math.max(a.distanceTo(b), 1e-6);
    return new THREE.CylinderGeometry(0.06, 0.06, len, 24);
  }
  if (n === 3) {
    const g = new THREE.BufferGeometry();
    const p = unique;
    const pos = new Float32Array([
      p[0].x,
      p[0].y,
      p[0].z,
      p[1].x,
      p[1].y,
      p[1].z,
      p[2].x,
      p[2].y,
      p[2].z,
    ]);
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setIndex([0, 1, 2]);
    g.computeVertexNormals();
    return g;
  }

  const tryHull = (pts: THREE.Vector3[]) => new ConvexGeometry(pts);

  try {
    return tryHull(unique);
  } catch {
    const bumped = unique.map((v) => v.clone());
    bumped[0].add(new THREE.Vector3(2e-4, -1e-4, 1.5e-4));
    try {
      return tryHull(dedupeVectors(bumped));
    } catch {
      return null;
    }
  }
}

function makeBackgroundTexture(): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 4;
  canvas.height = 512;
  const ctx = canvas.getContext("2d")!;
  const g = ctx.createLinearGradient(0, 0, 0, 512);
  g.addColorStop(0, "#0c1528");
  g.addColorStop(0.45, "#060a12");
  g.addColorStop(1, "#020408");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 4, 512);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

type Props = {
  frame: MorphFrame3D | null;
  frameLabel: string;
  /** Canvas height in px (match AgentTopology in side-by-side layout). */
  plotHeight?: number;
  /** After the final 3×αβγ cycle, color the newest spoke + node red. */
  highlightLastVectorRed?: boolean;
};

/** Persist orbit view across morph steps (same effect re-runs when `frame` changes). */
const defaultView = {
  position: new THREE.Vector3(2.35, 1.85, 2.55),
  target: new THREE.Vector3(0, 0.05, 0),
};

export function VectorSpace3D({
  frame,
  frameLabel,
  plotHeight = 320,
  highlightLastVectorRed = false,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const savedViewRef = useRef(defaultView);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !frame?.points?.length) return;

    let w = el.clientWidth || 480;
    let h = el.clientHeight || plotHeight;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x03060c, 0.11);
    scene.background = makeBackgroundTexture();

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.05, 120);
    camera.position.copy(savedViewRef.current.position);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    el.innerHTML = "";
    el.appendChild(renderer.domElement);

    const composer = new EffectComposer(renderer);
    const renderPass = new RenderPass(scene, camera);
    composer.addPass(renderPass);
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(w, h),
      0.35,
      0.6,
      0.88,
    );
    composer.addPass(bloomPass);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.055;
    controls.target.copy(savedViewRef.current.target);
    controls.maxPolarAngle = Math.PI * 0.92;
    controls.update();

    scene.add(new THREE.AmbientLight(0x6688cc, 0.28));
    scene.add(new THREE.HemisphereLight(0x7ab0ff, 0x1a1020, 0.55));

    const key = new THREE.DirectionalLight(0xffffff, 1.05);
    key.position.set(5, 8, 6);
    key.castShadow = false;
    scene.add(key);

    const rim = new THREE.DirectionalLight(0x4488ff, 0.45);
    rim.position.set(-5, 3, -6);
    scene.add(rim);

    const pt1 = new THREE.PointLight(0x66aaff, 1.2, 12, 2);
    pt1.position.set(2.8, 2.2, 1.5);
    scene.add(pt1);

    const pt2 = new THREE.PointLight(0xff9966, 0.85, 10, 2);
    pt2.position.set(-2.5, 1.2, -2.8);
    scene.add(pt2);

    const axes = new THREE.AxesHelper(1.25);
    axes.setColors(
      new THREE.Color(0xff4466),
      new THREE.Color(0x44ff88),
      new THREE.Color(0x4488ff),
    );
    scene.add(axes);

    const grid = new THREE.GridHelper(4, 20, 0x3a4f70, 0x151c28);
    grid.position.y = -1.35;
    (grid.material as THREE.Material).opacity = 0.45;
    (grid.material as THREE.Material).transparent = true;
    scene.add(grid);

    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.85, 0.012, 12, 64),
      new THREE.MeshBasicMaterial({
        color: 0x4a7ab8,
        transparent: true,
        opacity: 0.35,
      }),
    );
    ring.rotation.x = Math.PI / 2;
    ring.position.y = -1.32;
    scene.add(ring);

    const norm = normalize(frame.points);
    const rawVecs = norm.map((p) => new THREE.Vector3(p.x, p.y, p.z));
    const lastIdx = rawVecs.length - 1;
    const redLast = highlightLastVectorRed && lastIdx >= 0;
    /** Final corrupted anchor: stick out strongly toward +++ (normalized space ~±1 ball). */
    if (redLast) {
      rawVecs[lastIdx]!.add(new THREE.Vector3(1.35, 1.35, 1.35));
    }
    const unique = dedupeVectors(rawVecs);

    const hullGroup = new THREE.Group();
    scene.add(hullGroup);

    const disposables: THREE.BufferGeometry[] = [];
    const materials: THREE.Material[] = [];
    const extraDisposables: { dispose: () => void }[] = [];

    const tensorLinesGroup = new THREE.Group();
    hullGroup.add(tensorLinesGroup);

    const lineMatCyan = new THREE.LineBasicMaterial({
      color: 0x66ffe8,
      transparent: true,
      opacity: 1,
    });
    materials.push(lineMatCyan);
    const lineMatRed = new THREE.LineBasicMaterial({
      color: 0xff4466,
      transparent: true,
      opacity: 1,
    });
    materials.push(lineMatRed);

    const nodeMatCyan = new THREE.MeshBasicMaterial({ color: 0x99ffee });
    materials.push(nodeMatCyan);
    const nodeMatRed = new THREE.MeshBasicMaterial({ color: 0xff5566 });
    materials.push(nodeMatRed);

    const origin = new THREE.Vector3(0, 0, 0);
    const lineData: {
      geo: THREE.BufferGeometry;
      p: THREE.Vector3;
      phase: number;
      speed: number;
    }[] = [];

    for (let i = 0; i < rawVecs.length; i++) {
      const p = rawVecs[i]!;
      const isRed = redLast && i === lastIdx;
      const geo = new THREE.BufferGeometry().setFromPoints([origin, p]);
      const line = new THREE.Line(
        geo,
        isRed ? lineMatRed : lineMatCyan,
      );
      tensorLinesGroup.add(line);
      lineData.push({
        geo,
        p: p.clone(),
        phase: Math.random() * Math.PI * 2,
        speed: 1.5 + Math.random() * 2.0,
      });
      disposables.push(geo);

      const nodeGeo = new THREE.BoxGeometry(0.04, 0.04, 0.04);
      const nodeMesh = new THREE.Mesh(
        nodeGeo,
        isRed ? nodeMatRed : nodeMatCyan,
      );
      nodeMesh.position.copy(p);
      line.add(nodeMesh);
      disposables.push(nodeGeo);
    }

    const solidGeom = buildConvexSolidGeometry(unique);
    if (solidGeom) {
      const isBigHull = unique.length >= 4;

      const faceMat = new THREE.MeshPhysicalMaterial({
        color: 0x55eeff,
        emissive: 0x00ccb3,
        emissiveIntensity: 0.55,
        metalness: 0.35,
        roughness: 0.22,
        transparent: true,
        opacity: isBigHull ? 0.48 : 0.62,
        side: THREE.DoubleSide,
        depthWrite: !isBigHull,
        transmission: 0.22,
      });
      materials.push(faceMat);

      const solidMesh = new THREE.Mesh(solidGeom, faceMat);

      if (unique.length === 2) {
        const a = unique[0];
        const b = unique[1];
        const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
        solidMesh.position.copy(mid);
        const dir3 = new THREE.Vector3().subVectors(b, a).normalize();
        solidMesh.quaternion.setFromUnitVectors(
          new THREE.Vector3(0, 1, 0),
          dir3,
        );
      }

      hullGroup.add(solidMesh);
      disposables.push(solidGeom);

      if (isBigHull && solidGeom instanceof THREE.BufferGeometry) {
        try {
          const wireGeo = new THREE.WireframeGeometry(solidGeom);
          const wireMat = new THREE.LineBasicMaterial({
            color: 0x66ffee,
            transparent: true,
            opacity: 0.92,
          });
          materials.push(wireMat);
          const wireframe = new THREE.LineSegments(wireGeo, wireMat);
          wireframe.position.copy(solidMesh.position);
          wireframe.quaternion.copy(solidMesh.quaternion);
          hullGroup.add(wireframe);
          disposables.push(wireGeo);
        } catch {
          /* ignore */
        }
      }
    }

    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      
      const t = performance.now() * 0.001;
      
      // Global orbit
      hullGroup.rotation.y = t * 0.2;
      hullGroup.rotation.x = Math.sin(t * 0.1) * 0.15;
      
      // Breathing scaling
      const globalScale = 1 + 0.04 * Math.sin(t * 3);
      hullGroup.scale.set(globalScale, globalScale, globalScale);

      // Animate each independent vector stretching based on differential eq simulations
      for (let i = 0; i < lineData.length; i++) {
        const { geo, p, phase, speed } = lineData[i];
        const positions = geo.attributes.position.array as Float32Array;
        
        // Stretch simulates the deformation and decay of vectors over time
        const stretch = 1 + 0.25 * Math.sin(t * speed + phase);
        
        positions[3] = p.x * stretch;
        positions[4] = p.y * stretch;
        positions[5] = p.z * stretch;
        geo.attributes.position.needsUpdate = true;
        
        // Update the child box node position
        const lineMesh = tensorLinesGroup.children[i] as THREE.Line;
        const nodeMesh = lineMesh.children[0] as THREE.Mesh | undefined;
        nodeMesh?.position.set(positions[3], positions[4], positions[5]);
      }

      controls.update();
      composer.render();
    };
    tick();

    extraDisposables.push({
      dispose: () => {
        composer.dispose();
        renderPass.dispose?.();
        bloomPass.dispose?.();
      },
    });

    const onResize = () => {
      if (!el) return;
      w = el.clientWidth || 480;
      h = el.clientHeight || plotHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      composer.setSize(w, h);
      bloomPass.setSize(w, h);
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(el);

    return () => {
      savedViewRef.current.position.copy(camera.position);
      savedViewRef.current.target.copy(controls.target);
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      disposables.forEach((g) => g.dispose());
      materials.forEach((m) => m.dispose());
      ring.geometry.dispose();
      (ring.material as THREE.Material).dispose();
      axes.dispose();
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      const bg = scene.background;
      if (bg instanceof THREE.CanvasTexture) bg.dispose();
      extraDisposables.forEach((x) => x.dispose());
      renderer.dispose();
      if (renderer.domElement.parentNode === el) {
        el.removeChild(renderer.domElement);
      }
    };
  }, [frame, plotHeight, highlightLastVectorRed]);

  if (!frame?.points?.length) {
    return (
      <div
        className="viz-3d-empty"
        style={{
          height: plotHeight,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "#5c6d8a",
          fontSize: 13,
        }}
      >
        No points
      </div>
    );
  }

  return (
    <div className="viz-3d-root">
      <div
        ref={containerRef}
        style={{ width: "100%", height: plotHeight, minHeight: plotHeight }}
      />
      <p
        className="viz-3d-caption"
        style={{
          margin: "0.5rem 0 0",
          fontSize: "0.78rem",
          color: "#8b9cb8",
          lineHeight: 1.45,
        }}
      >
        {frameLabel} · <strong>Convex hull</strong> of all{" "}
        <code style={{ color: "#5b9cfa" }}>W·x</code> vectors ({frame.embed_dim}
        D), 3 PCA axes, inner shell + wireframe, bloom. Drag to orbit.
      </p>
    </div>
  );
}
