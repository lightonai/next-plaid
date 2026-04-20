import init, {
  handle_storage_request_json,
  handle_runtime_request_json,
  reset_runtime_state
} from "./pkg/next_plaid_browser_wasm.js";

let runtimeReady;
const storageRequestTypes = new Set(["install_bundle", "load_stored_bundle"]);

async function ensureRuntime() {
  if (!runtimeReady) {
    runtimeReady = init().then(() => {
      reset_runtime_state();
    });
  }

  await runtimeReady;
}

async function handleRequest(request) {
  await ensureRuntime();
  const responseJson = storageRequestTypes.has(request?.type)
    ? await handle_storage_request_json(JSON.stringify(request))
    : handle_runtime_request_json(JSON.stringify(request));
  return JSON.parse(responseJson);
}

self.addEventListener("message", async (event) => {
  const { requestId, request } = event.data ?? {};

  if (typeof requestId !== "string") {
    self.postMessage({
      requestId: null,
      ok: false,
      error: "Worker request envelope must include a string requestId"
    });
    return;
  }

  if (!request || typeof request !== "object") {
    self.postMessage({
      requestId,
      ok: false,
      error: "Worker request envelope must include an object request payload"
    });
    return;
  }

  try {
    const response = await handleRequest(request);
    self.postMessage({ requestId, ok: true, response });
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    self.postMessage({ requestId, ok: false, error: message });
  }
});
