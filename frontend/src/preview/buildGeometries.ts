import * as THREE from "three";

import type {
  BandEdgeLine,
  BandShell,
  PreviewGroundPlane,
  PreviewMesh,
  PreviewPoses,
} from "../types";

/** Build indexed mesh geometry from API positions/indices. */
export function buildObjectGeometry(mesh: PreviewMesh): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(mesh.positions, 3),
  );
  geometry.setIndex(mesh.indices);
  geometry.computeVertexNormals();
  return geometry;
}

/** Triangulate the band surface grid (elevation-major rows, azimuth columns). */
export function buildBandSurfaceIndices(
  azimuthCount: number,
  elevationCount: number,
): number[] {
  const indices: number[] = [];
  for (let row = 0; row < elevationCount - 1; row += 1) {
    for (let col = 0; col < azimuthCount - 1; col += 1) {
      const topLeft = row * azimuthCount + col;
      const topRight = topLeft + 1;
      const bottomLeft = (row + 1) * azimuthCount + col;
      const bottomRight = bottomLeft + 1;
      indices.push(topLeft, bottomLeft, topRight);
      indices.push(topRight, bottomLeft, bottomRight);
    }
  }
  return indices;
}

export function buildBandSurfaceGeometry(shell: BandShell): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(shell.positions, 3),
  );
  geometry.setIndex(
    buildBandSurfaceIndices(shell.azimuth_count, shell.elevation_count),
  );
  geometry.computeVertexNormals();
  return geometry;
}

export function buildBandEdgeGeometry(edge: BandEdgeLine): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(edge.positions, 3),
  );
  return geometry;
}

export function buildGroundPlaneGeometry(
  ground: PreviewGroundPlane,
): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(ground.positions, 3),
  );
  geometry.setIndex(ground.indices);
  geometry.computeVertexNormals();
  return geometry;
}

export function buildRayGeometry(
  rays: [THREE.Vector3Tuple, THREE.Vector3Tuple][],
): THREE.BufferGeometry {
  const positions: number[] = [];
  for (const [start, end] of rays) {
    positions.push(start[0], start[1], start[2], end[0], end[1], end[2]);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  return geometry;
}

/** Suggested camera distance from mesh bbox extent (display framing only). */
export function bboxViewDistance(mesh: PreviewMesh): number {
  const [[minX, minY, minZ], [maxX, maxY, maxZ]] = mesh.bbox;
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const extent = Math.max(dx, dy, dz, 0.1);
  return extent * 2.8;
}

export type PreviewSceneObjects = {
  bandSurface: THREE.Mesh | null;
  bandEdges: THREE.Line[];
  groundPlane: THREE.Mesh | null;
  cameraMarkers: THREE.InstancedMesh | null;
  rays: THREE.LineSegments | null;
};

export function buildPreviewObjects(poses: PreviewPoses | null): PreviewSceneObjects {
  if (poses === null) {
    return {
      bandSurface: null,
      bandEdges: [],
      groundPlane: null,
      cameraMarkers: null,
      rays: null,
    };
  }

  const bandSurfaceGeom = buildBandSurfaceGeometry(poses.band.surface);
  const bandSurface = new THREE.Mesh(
    bandSurfaceGeom,
    new THREE.MeshStandardMaterial({
      color: 0x58a6ff,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );

  const bandEdges = poses.band.edges.map((edge) => {
    const geometry = buildBandEdgeGeometry(edge);
    const material = new THREE.LineBasicMaterial({
      color: edge.kind === "azimuth" ? 0x79c0ff : 0xa371f7,
      transparent: true,
      opacity: 0.85,
    });
    return new THREE.Line(geometry, material);
  });

  const groundGeom = buildGroundPlaneGeometry(poses.ground_plane);
  const groundPlane = new THREE.Mesh(
    groundGeom,
    new THREE.MeshBasicMaterial({
      color: 0x3fb950,
      transparent: true,
      opacity: 0.22,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );

  const locations = poses.cameras.locations;
  let cameraMarkers: THREE.InstancedMesh | null = null;
  if (locations.length > 0) {
    const markerRadius = Math.max(0.04, poses.display_radius * 0.01);
    const markerGeometry = new THREE.SphereGeometry(markerRadius, 10, 10);
    const markerMaterial = new THREE.MeshStandardMaterial({ color: 0xf0883e });
    const instanced = new THREE.InstancedMesh(
      markerGeometry,
      markerMaterial,
      locations.length,
    );
    const matrix = new THREE.Matrix4();
    locations.forEach((location, index) => {
      matrix.setPosition(location[0], location[1], location[2]);
      instanced.setMatrixAt(index, matrix);
    });
    instanced.instanceMatrix.needsUpdate = true;
    cameraMarkers = instanced;
  }

  const rayGeometry = buildRayGeometry(poses.cameras.rays);
  const rays = new THREE.LineSegments(
    rayGeometry,
    new THREE.LineBasicMaterial({
      color: 0xff7b72,
      transparent: true,
      opacity: 0.55,
    }),
  );

  return {
    bandSurface,
    bandEdges,
    groundPlane,
    cameraMarkers,
    rays,
  };
}

export function disposeObject3D(object: THREE.Object3D): void {
  object.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Line) {
      child.geometry.dispose();
      const material = child.material;
      if (Array.isArray(material)) {
        material.forEach((entry) => entry.dispose());
      } else {
        material.dispose();
      }
    }
  });
}
