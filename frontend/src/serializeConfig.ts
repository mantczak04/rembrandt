import { createDefaultConfig } from "./defaultConfig";
import type { RembrandtConfig } from "./types";

/** Merge user edits with schema defaults for an honest save preview. */
export function mergeConfigDefaults(config: RembrandtConfig): RembrandtConfig {
  const defaults = createDefaultConfig(config.object.path);
  return {
    ...defaults,
    ...config,
    object: { ...defaults.object, ...config.object },
    camera: { ...defaults.camera, ...config.camera },
    lights: config.lights ?? defaults.lights,
    render: { ...defaults.render, ...config.render },
    output: { ...defaults.output, ...config.output },
    background: { ...defaults.background, ...config.background },
    light_randomization: {
      ...defaults.light_randomization,
      ...config.light_randomization,
    },
    labels: { ...defaults.labels, ...config.labels },
    framing: { ...defaults.framing, ...config.framing },
    postfx: { ...defaults.postfx, ...config.postfx },
  };
}

/** Serialize the config object as YAML for display only (no domain logic). */
export function configToYaml(config: RembrandtConfig): string {
  return serializeYaml(mergeConfigDefaults(config));
}

function serializeYaml(value: unknown, indent = 0): string {
  const pad = " ".repeat(indent);

  if (value === null || value === undefined) {
    return "null";
  }

  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }

  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : String(value);
  }

  if (typeof value === "string") {
    if (needsQuotes(value)) {
      return JSON.stringify(value);
    }
    return value;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "[]";
    }
    const lines = value.map((item) => {
      const serialized = serializeYaml(item, indent + 2);
      if (isPlainObject(item)) {
        const inner = serialized
          .split("\n")
          .map((line) => (line ? `${pad}  ${line}` : line))
          .join("\n");
        return `${pad}- ${inner.trimStart()}`;
      }
      return `${pad}- ${serialized}`;
    });
    return lines.join("\n");
  }

  if (isPlainObject(value)) {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return "{}";
    }
    return entries
      .map(([key, child]) => {
        const childYaml = serializeYaml(child, indent + 2);
        if (isMultilineYaml(childYaml)) {
          return `${pad}${key}:\n${childYaml}`;
        }
        return `${pad}${key}: ${childYaml}`;
      })
      .join("\n");
  }

  return JSON.stringify(value);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isMultilineYaml(yaml: string): boolean {
  return yaml.includes("\n");
}

function needsQuotes(value: string): boolean {
  if (value === "") {
    return true;
  }
  if (/^\s|\s$/.test(value)) {
    return true;
  }
  if (/[:#{}[\],&*?|>-]/.test(value)) {
    return true;
  }
  if (/^(true|false|null|yes|no|on|off)$/i.test(value)) {
    return true;
  }
  return false;
}
