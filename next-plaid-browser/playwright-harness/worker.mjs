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

self.addEventListener("message", async (event) => {
  const { requestId, request } = event.data ?? {};

  try {
    await ensureRuntime();
    const responseJson = handle_runtime_request_json(JSON.stringify(request));
    const response = JSON.parse(responseJson);
    self.postMessage({ requestId, ok: true, response });
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    self.postMessage({ requestId, ok: false, error: message });
  }
});
