import type {
  HealthResponse,
  PreviewMesh,
  PreviewPoses,
  PreviewPosesParams,
  RembrandtConfig,
  SaveConfigResponse,
  SourceUpAxis,
} from "./types";

const API_BASE = "/api";
const DEFAULT_POSES_DEBOUNCE_MS = 100;

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type PendingPosesRequest = {
  timer: ReturnType<typeof setTimeout> | null;
  reject: (reason: unknown) => void;
  controller: AbortController;
};

let pendingPosesRequest: PendingPosesRequest | null = null;

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE}/health`);
  return parseJsonResponse<HealthResponse>(response);
}

export async function fetchMesh(
  path: string,
  upAxis: SourceUpAxis = "Z",
  normalize: boolean = true,
): Promise<PreviewMesh> {
  return postJson<PreviewMesh>("/preview/mesh", {
    path,
    up_axis: upAxis,
    normalize,
  });
}

export function fetchPoses(
  params: PreviewPosesParams,
  debounceMs: number = DEFAULT_POSES_DEBOUNCE_MS,
): Promise<PreviewPoses> {
  cancelPendingPosesRequest();

  return new Promise((resolve, reject) => {
    const controller = new AbortController();
    const pending: PendingPosesRequest = {
      timer: null,
      reject,
      controller,
    };

    pending.timer = setTimeout(() => {
      pending.timer = null;
      void fetchPosesNow(params, { signal: controller.signal })
        .then(resolve)
        .catch(reject)
        .finally(() => {
          if (pendingPosesRequest === pending) {
            pendingPosesRequest = null;
          }
        });
    }, debounceMs);

    pendingPosesRequest = pending;
  });
}

export async function fetchPosesNow(
  params: PreviewPosesParams,
  init: RequestInit = {},
): Promise<PreviewPoses> {
  return postJson<PreviewPoses>("/preview/poses", params, init);
}

export async function fetchConfigDefaults(): Promise<RembrandtConfig> {
  const response = await fetch(`${API_BASE}/config/defaults`);
  return parseJsonResponse<RembrandtConfig>(response);
}

export async function saveConfig(
  config: RembrandtConfig,
  filename: string,
): Promise<SaveConfigResponse> {
  return postJson<SaveConfigResponse>("/config/save", { config, filename });
}

async function postJson<T>(
  path: string,
  body: unknown,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseJsonResponse<T>(response);
}

function cancelPendingPosesRequest(): void {
  if (pendingPosesRequest === null) {
    return;
  }

  if (pendingPosesRequest.timer !== null) {
    clearTimeout(pendingPosesRequest.timer);
  }
  pendingPosesRequest.controller.abort();
  pendingPosesRequest.reject(abortError());
  pendingPosesRequest = null;
}

function abortError(): Error {
  const error = new Error("Preview pose request was superseded by a newer update");
  error.name = "AbortError";
  return error;
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw await errorFromResponse(response);
  }
  return (await response.json()) as T;
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  let message = `Request failed (${response.status})`;
  try {
    const payload = (await response.json()) as { detail?: unknown };
    if (typeof payload.detail === "string") {
      message = payload.detail;
    } else if (Array.isArray(payload.detail)) {
      message = formatValidationDetail(payload.detail);
    }
  } catch {
    // Response body was not JSON.
  }
  return new ApiError(message, response.status);
}

function formatValidationDetail(detail: unknown[]): string {
  const messages = detail
    .map((entry) => {
      if (!isValidationError(entry)) {
        return null;
      }
      const location = entry.loc
        .map((part) => String(part))
        .filter((part) => part !== "body")
        .join(".");
      return location ? `${location}: ${entry.msg}` : entry.msg;
    })
    .filter((entry): entry is string => entry !== null);

  return messages.length > 0 ? messages.join("; ") : "Request validation failed";
}

function isValidationError(
  value: unknown,
): value is { loc: Array<string | number>; msg: string } {
  return (
    typeof value === "object" &&
    value !== null &&
    "loc" in value &&
    "msg" in value &&
    Array.isArray(value.loc) &&
    typeof value.msg === "string"
  );
}
