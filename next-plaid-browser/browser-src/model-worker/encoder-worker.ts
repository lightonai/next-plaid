import type {
  EncoderBackend,
  EncoderHealthResponse,
  EncoderInitResponse,
  EncoderState,
  EncoderWorkerRequest,
  EncoderWorkerResponse,
} from "./types.js";
import type {
  WorkerEventEnvelope,
  WorkerFailureEnvelope,
  WorkerRequestEnvelope,
  WorkerSuccessEnvelope,
} from "../shared/worker-envelope.js";
import { createWasmEncoderBackend } from "./wasm-encoder-backend.js";

let backend: EncoderBackend | null = null;
let state: EncoderState = "empty";
let lastError: string | null = null;

function postFrame(payload: unknown): void {
  self.postMessage([1, payload]);
}

function postSuccess(
  requestId: string,
  response: EncoderWorkerResponse,
): void {
  const envelope: WorkerSuccessEnvelope<EncoderWorkerResponse> = {
    requestId,
    ok: true,
    response,
  };
  postFrame(envelope);
}

function postEvent(requestId: string, event: unknown): void {
  const envelope: WorkerEventEnvelope<unknown> = {
    requestId,
    ok: true,
    event,
  };
  postFrame(envelope);
}

function postFailure(requestId: string | null, error: string): void {
  const envelope: WorkerFailureEnvelope = {
    requestId: requestId ?? "",
    ok: false,
    error,
  };
  postFrame(envelope);
}

self.postMessage([0]);

async function handleRequest(
  requestId: string,
  request: EncoderWorkerRequest,
): Promise<EncoderWorkerResponse> {
  switch (request.type) {
    case "init": {
      if (backend) {
        await backend.dispose();
        backend = null;
      }
      state = "initializing";
      lastError = null;
      backend = await createWasmEncoderBackend(request.payload, (event) => {
        postEvent(requestId, event);
      });
      state = "ready";
      const response: EncoderInitResponse = {
        type: "encoder_ready",
        state: "ready",
        capabilities: backend.capabilities,
      };
      return response;
    }
    case "health": {
      const response: EncoderHealthResponse = {
        type: "encoder_health",
        state,
        health: backend ? await backend.health() : "degraded",
        capabilities: backend?.capabilities ?? null,
        last_error: lastError,
      };
      return response;
    }
    case "encode": {
      if (!backend || state !== "ready") {
        throw new Error("encoder worker is not ready");
      }
      return {
        type: "encoded_query",
        encoded: await backend.encode(request.payload.text),
      };
    }
    case "dispose": {
      if (backend) {
        await backend.dispose();
        backend = null;
      }
      state = "disposed";
      return {
        type: "encoder_disposed",
      };
    }
    default: {
      const impossible: never = request;
      throw new Error(`unsupported encoder request ${(impossible as { type?: string }).type ?? "unknown"}`);
    }
  }
}

self.addEventListener("message", async (event: MessageEvent<unknown>) => {
  const frame = event.data as unknown;
  if (Array.isArray(frame) && frame.length === 1 && frame[0] === 1) {
    self.close();
    return;
  }
  if (!Array.isArray(frame) || frame[0] !== 0) {
    return;
  }
  const data = (frame[1] ?? {}) as Partial<WorkerRequestEnvelope<EncoderWorkerRequest>>;
  const requestId = typeof data.requestId === "string" ? data.requestId : null;
  const request = data.request;

  if (typeof requestId !== "string") {
    postFailure(null, "Worker request envelope must include a string requestId");
    return;
  }
  if (!request || typeof request !== "object") {
    postFailure(requestId, "Worker request envelope must include an object request payload");
    return;
  }

  try {
    const response = await handleRequest(requestId, request);
    postSuccess(requestId, response);
  } catch (error) {
    state = "failed";
    lastError = error instanceof Error ? error.stack ?? error.message : String(error);
    postFailure(requestId, lastError);
  }
});
