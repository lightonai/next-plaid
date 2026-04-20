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

function postFrame(payload) {
  self.postMessage([1, payload]);
}

self.postMessage([0]);

self.addEventListener("message", async (event) => {
  const frame = event.data;
  if (Array.isArray(frame) && frame.length === 1 && frame[0] === 1) {
    self.close();
    return;
  }
  if (!Array.isArray(frame) || frame[0] !== 0) {
    return;
  }
  const data = frame[1] ?? {};
  const { requestId, request } = data;

  if (typeof requestId !== "string") {
    postFrame({
      requestId: "",
      ok: false,
      error: "Worker request envelope must include a string requestId"
    });
    return;
  }

  if (!request || typeof request !== "object") {
    postFrame({
      requestId,
      ok: false,
      error: "Worker request envelope must include an object request payload"
    });
    return;
  }

  try {
    const response = await handleRequest(request);
    postFrame({ requestId, ok: true, response });
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    postFrame({ requestId, ok: false, error: message });
  }
});
