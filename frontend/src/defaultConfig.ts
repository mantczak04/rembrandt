import type { RembrandtConfig } from "./types";

/** Overlay user object path onto schema defaults fetched from the API. */
export function createDefaultConfig(
  objectPath: string,
  schemaDefaults: RembrandtConfig,
): RembrandtConfig {
  return {
    ...schemaDefaults,
    object: { ...schemaDefaults.object, path: objectPath },
  };
}
