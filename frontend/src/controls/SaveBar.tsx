import { useState } from "react";

import { ApiError, saveConfig } from "../api";
import { configToYaml } from "../serializeConfig";
import type { RembrandtConfig } from "../types";
import styles from "./Controls.module.css";

export type SaveBarProps = {
  config: RembrandtConfig;
  schemaDefaults: RembrandtConfig | null;
  disabled?: boolean;
};

function normalizeFilename(filename: string): string {
  const trimmed = filename.trim();
  if (!trimmed) {
    return "dataset.yaml";
  }
  return trimmed.endsWith(".yaml") || trimmed.endsWith(".yml")
    ? trimmed
    : `${trimmed}.yaml`;
}

export default function SaveBar({
  config,
  schemaDefaults,
  disabled = false,
}: SaveBarProps) {
  const [filename, setFilename] = useState("dataset.yaml");
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showPreview, setShowPreview] = useState(true);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    setSavedPath(null);
    try {
      const response = await saveConfig(config, normalizeFilename(filename));
      setSavedPath(response.path);
    } catch (saveError) {
      if (saveError instanceof ApiError) {
        setError(saveError.message);
      } else if (saveError instanceof Error) {
        setError(saveError.message);
      } else {
        setError("Failed to save config");
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <section className={styles.panel}>
      <h2 className={styles.sectionTitle}>Save config</h2>
      <div className={styles.row}>
        <label className={styles.label} htmlFor="config-filename">
          Filename
        </label>
        <input
          id="config-filename"
          className={styles.input}
          type="text"
          value={filename}
          onChange={(event) => setFilename(event.target.value)}
          placeholder="dataset.yaml"
        />
        <button
          type="button"
          className={styles.button}
          disabled={
            disabled || schemaDefaults === null || saving || config.object.path.trim() === ""
          }
          onClick={() => void handleSave()}
        >
          {saving ? "Saving…" : "Save"}
        </button>
        {savedPath ? (
          <p className={styles.hint}>
            Saved to <code>{savedPath}</code>
          </p>
        ) : null}
        {error ? <p className={styles.error}>{error}</p> : null}
      </div>

      <label className={styles.checkboxRow}>
        <input
          type="checkbox"
          checked={showPreview}
          onChange={(event) => setShowPreview(event.target.checked)}
        />
        Show YAML preview
      </label>

      {showPreview ? (
        <pre className={styles.yamlPreview} aria-label="Config YAML preview">
          {schemaDefaults ? configToYaml(config, schemaDefaults) : "Loading defaults…"}
        </pre>
      ) : null}
    </section>
  );
}
