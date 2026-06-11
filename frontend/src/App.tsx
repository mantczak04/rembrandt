import { useCallback, useEffect, useState } from "react";

import { ApiError, fetchHealth, fetchMesh, fetchPoses } from "./api";
import Controls from "./controls/Controls";
import SaveBar from "./controls/SaveBar";
import { createDefaultConfig } from "./defaultConfig";
import Viewport from "./preview/Viewport";
import type {
  CameraConfig,
  PreviewMesh,
  PreviewPoses,
  PreviewPosesParams,
  RembrandtConfig,
  SourceUpAxis,
} from "./types";
import styles from "./App.module.css";

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

function posesParams(mesh: PreviewMesh, camera: CameraConfig): PreviewPosesParams {
  return {
    bbox: mesh.bbox,
    n: camera.n,
    azimuth_range: camera.azimuth_range,
    elevation_range: camera.elevation_range,
    distance_range: camera.distance_range,
    strategy: camera.strategy,
    seed: camera.seed,
    look_at: camera.look_at,
  };
}

export default function App() {
  const [config, setConfig] = useState<RembrandtConfig>(() => createDefaultConfig());
  const [objectPathInput, setObjectPathInput] = useState("");
  const [mesh, setMesh] = useState<PreviewMesh | null>(null);
  const [poses, setPoses] = useState<PreviewPoses | null>(null);
  const [showCameras, setShowCameras] = useState(true);
  const [showRays, setShowRays] = useState(true);
  const [loadingMesh, setLoadingMesh] = useState(false);
  const [loadingPoses, setLoadingPoses] = useState(false);
  const [meshError, setMeshError] = useState<string | null>(null);
  const [posesError, setPosesError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void fetchHealth()
      .then((payload) => {
        if (!cancelled) {
          setApiStatus(payload.status);
          setApiError(null);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setApiStatus(null);
          if (error instanceof ApiError) {
            setApiError(error.message);
          } else if (error instanceof Error) {
            setApiError(error.message);
          } else {
            setApiError("Backend unreachable");
          }
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshPoses = useCallback(
    async (nextMesh: PreviewMesh, camera: CameraConfig) => {
      setLoadingPoses(true);
      setPosesError(null);
      try {
        const nextPoses = await fetchPoses(posesParams(nextMesh, camera));
        setPoses(nextPoses);
      } catch (error) {
        if (isAbortError(error)) {
          return;
        }
        setPoses(null);
        if (error instanceof ApiError) {
          setPosesError(error.message);
        } else if (error instanceof Error) {
          setPosesError(error.message);
        } else {
          setPosesError("Failed to load camera preview");
        }
      } finally {
        setLoadingPoses(false);
      }
    },
    [],
  );

  const handleLoadMesh = useCallback(async () => {
    const path = objectPathInput.trim();
    if (!path) {
      return;
    }

    const upAxis = config.object.up_axis ?? "Z";
    setLoadingMesh(true);
    setMeshError(null);
    setPoses(null);
    try {
      const nextMesh = await fetchMesh(path, upAxis);
      setMesh(nextMesh);
      let cameraForPoses = config.camera;
      setConfig((current) => {
        cameraForPoses = current.camera;
        return { ...current, object: { path, up_axis: upAxis } };
      });
      await refreshPoses(nextMesh, cameraForPoses);
    } catch (error) {
      setMesh(null);
      if (error instanceof ApiError) {
        setMeshError(error.message);
      } else if (error instanceof Error) {
        setMeshError(error.message);
      } else {
        setMeshError("Failed to load mesh");
      }
    } finally {
      setLoadingMesh(false);
    }
  }, [config.camera, config.object.up_axis, objectPathInput, refreshPoses]);

  const handleObjectUpAxisChange = useCallback((up_axis: SourceUpAxis) => {
    setConfig((current) => ({
      ...current,
      object: { ...current.object, up_axis },
    }));
    setMesh(null);
    setPoses(null);
  }, []);

  const handleCameraChange = useCallback(
    (camera: CameraConfig) => {
      setConfig((current) => ({ ...current, camera }));
      if (mesh) {
        void refreshPoses(mesh, camera);
      }
    },
    [mesh, refreshPoses],
  );

  const handleConfigChange = useCallback((nextConfig: RembrandtConfig) => {
    setConfig(nextConfig);
  }, []);

  return (
    <div className={styles.layout}>
      <header className={styles.header}>
        <h1 className={styles.eyebrow}>Rembrandt</h1>
        <p className={styles.subtitle}>
          Tune camera-sphere coverage on your object, then save a YAML config for{" "}
          <code>rembrandt-render</code>.
        </p>
        <p className={styles.statusLine}>
          API:{" "}
          {apiStatus ? (
            <span className={styles.statusOk}>connected ({apiStatus})</span>
          ) : apiError ? (
            <span className={styles.statusError}>{apiError}</span>
          ) : (
            "checking…"
          )}
        </p>
      </header>

      <div className={styles.main}>
        <div className={styles.viewportPane}>
          <Viewport
            mesh={mesh}
            poses={poses}
            showCameras={showCameras}
            showRays={showRays}
          />
        </div>
        <div className={styles.controlsPane}>
          <Controls
            config={config}
            objectPathInput={objectPathInput}
            showCameras={showCameras}
            showRays={showRays}
            loadingMesh={loadingMesh}
            loadingPoses={loadingPoses}
            meshError={meshError}
            posesError={posesError}
            onObjectPathInputChange={setObjectPathInput}
            onObjectUpAxisChange={handleObjectUpAxisChange}
            onLoadMesh={() => void handleLoadMesh()}
            onCameraChange={handleCameraChange}
            onConfigChange={handleConfigChange}
            onShowCamerasChange={setShowCameras}
            onShowRaysChange={setShowRays}
          />
        </div>
      </div>

      <div className={styles.savePane}>
        <SaveBar config={config} disabled={mesh === null} />
      </div>
    </div>
  );
}
