import { useState } from "react";

import { ApiError, saveConfig } from "../api";
import type { RembrandtConfig } from "../types";
import styles from "./Controls.module.css";

export type SaveBarProps = {
  config: RembrandtConfig;
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

export default function SaveBar({ config, disabled = false }: SaveBarProps) {
  const [filename, setFilename] = useState("dataset.yaml");
  const [savedPath, setSavedPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [showPreview, setShowPreview] = useState(false);

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
          disabled={disabled || saving || config.object.path.trim() === ""}
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
        Show config preview
      </label>

      {showPreview ? (
        <pre
          style={{
            margin: 0,
            padding: "0.75rem",
            fontSize: "0.75rem",
            overflow: "auto",
            maxHeight: "14rem",
            borderRadius: "0.375rem",
            background: "#0d1117",
            border: "1px solid #30363d",
            color: "#c9d1d9",
          }}
        >
          {JSON.stringify(config, null, 2)}
        </pre>
      ) : null}
    </section>
  );
}
