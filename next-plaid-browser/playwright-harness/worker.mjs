import init, {
  handle_runtime_request_json,
  reset_runtime_state
} from "./pkg/next_plaid_browser_wasm.js";

let runtimeReady;

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
  const responseJson = handle_runtime_request_json(JSON.stringify(request));
  return JSON.parse(responseJson);
}

self.addEventListener("message", async (event) => {
  const { requestId, request } = event.data ?? {};

  try {
    const response = await handleRequest(request);
    self.postMessage({ requestId, ok: true, response });
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    self.postMessage({ requestId, ok: false, error: message });
  }
});
