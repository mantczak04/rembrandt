import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { PreviewMesh, PreviewPoses } from "../types";
import {
  bboxViewDistance,
  buildObjectGeometry,
  buildPreviewObjects,
  disposeObject3D,
} from "./buildGeometries";
import styles from "./Viewport.module.css";

export type ViewportProps = {
  mesh: PreviewMesh | null;
  poses: PreviewPoses | null;
  showCameras?: boolean;
  showRays?: boolean;
  className?: string;
};

type PreviewGroup = {
  objectMesh: THREE.Mesh | null;
  bandSurface: THREE.Mesh | null;
  bandEdges: THREE.Line[];
  groundPlane: THREE.Mesh | null;
  cameraMarkers: THREE.InstancedMesh | null;
  rays: THREE.LineSegments | null;
};

type SceneHandle = {
  rebuild: (mesh: PreviewMesh | null, poses: PreviewPoses | null) => void;
  setVisibility: (showCameras: boolean, showRays: boolean) => void;
};

const emptyPreviewGroup = (): PreviewGroup => ({
  objectMesh: null,
  bandSurface: null,
  bandEdges: [],
  groundPlane: null,
  cameraMarkers: null,
  rays: null,
});

function applyVisibility(
  group: PreviewGroup,
  showCameras: boolean,
  showRays: boolean,
): void {
  if (group.cameraMarkers) {
    group.cameraMarkers.visible = showCameras;
  }
  if (group.rays) {
    group.rays.visible = showRays;
  }
}

export default function Viewport({
  mesh,
  poses,
  showCameras = true,
  showRays = true,
  className,
}: ViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneHandleRef = useRef<SceneHandle | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x010409);

    const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 500);
    camera.up.set(0, 0, 1);
    camera.position.set(4, -4, 3);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);
    renderer.domElement.className = styles.canvas;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
    keyLight.position.set(4, 6, 8);
    scene.add(keyLight);
    const grid = new THREE.GridHelper(12, 24, 0x30363d, 0x21262d);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(1.5));

    const previewRoot = new THREE.Group();
    scene.add(previewRoot);

    let previewGroup = emptyPreviewGroup();
    let showCamerasState = showCameras;
    let showRaysState = showRays;

    const clearPreviewGroup = () => {
      for (const child of [...previewRoot.children]) {
        disposeObject3D(child);
        previewRoot.remove(child);
      }
      previewGroup = emptyPreviewGroup();
    };

    const frameCamera = (nextMesh: PreviewMesh) => {
      const distance = bboxViewDistance(nextMesh);
      controls.target.set(0, 0, 0);
      camera.position.set(distance * 0.85, -distance * 0.85, distance * 0.65);
      controls.update();
    };

    const rebuild = (nextMesh: PreviewMesh | null, nextPoses: PreviewPoses | null) => {
      clearPreviewGroup();

      if (nextMesh) {
        const objectMesh = new THREE.Mesh(
          buildObjectGeometry(nextMesh),
          new THREE.MeshStandardMaterial({
            color: 0x9da7b3,
            roughness: 0.75,
            metalness: 0.05,
          }),
        );
        previewRoot.add(objectMesh);
        previewGroup.objectMesh = objectMesh;
        frameCamera(nextMesh);
      }

      if (nextPoses) {
        const built = buildPreviewObjects(nextPoses);
        if (built.bandSurface) {
          previewRoot.add(built.bandSurface);
        }
        for (const edge of built.bandEdges) {
          previewRoot.add(edge);
        }
        if (built.groundPlane) {
          previewRoot.add(built.groundPlane);
        }
        if (built.cameraMarkers) {
          previewRoot.add(built.cameraMarkers);
        }
        if (built.rays) {
          previewRoot.add(built.rays);
        }
        previewGroup = { ...built, objectMesh: previewGroup.objectMesh };
        applyVisibility(previewGroup, showCamerasState, showRaysState);
      }
    };

    sceneHandleRef.current = {
      rebuild,
      setVisibility: (nextShowCameras, nextShowRays) => {
        showCamerasState = nextShowCameras;
        showRaysState = nextShowRays;
        applyVisibility(previewGroup, showCamerasState, showRaysState);
      },
    };

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width === 0 || height === 0) {
        return;
      }
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height, false);
    };

    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    let frameId = 0;
    const animate = () => {
      frameId = window.requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      sceneHandleRef.current = null;
      window.cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      clearPreviewGroup();
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  useEffect(() => {
    sceneHandleRef.current?.rebuild(mesh, poses);
  }, [mesh, poses]);

  useEffect(() => {
    sceneHandleRef.current?.setVisibility(showCameras, showRays);
  }, [showCameras, showRays]);

  const status =
    mesh === null
      ? "Load an OBJ path to preview"
      : poses === null
        ? "Loading camera band…"
        : `${poses.cameras.locations.length} camera poses`;

  return (
    <div
      ref={containerRef}
      className={[styles.viewport, className].filter(Boolean).join(" ")}
      aria-label="3D camera pose preview"
    >
      <p className={styles.overlay}>{status}</p>
    </div>
  );
}
