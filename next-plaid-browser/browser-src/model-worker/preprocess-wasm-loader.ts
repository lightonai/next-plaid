import { Effect } from "effect";

import {
  type WorkerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";

export interface BrowserPreprocessWasmModule {
  readonly default: () => Promise<unknown>;
  readonly init: (
    tokenizerJsonBytes: Uint8Array,
    onnxConfigJsonBytes: Uint8Array,
  ) => unknown;
  readonly prepare_query: (text: string) => unknown;
  readonly prepare_document: (text: string) => unknown;
  readonly reset: () => void;
}

const preprocessWasmModuleUrl = new URL(
  "../../playwright-harness/preprocess-pkg/next_plaid_browser_preprocess_wasm.js",
  import.meta.url,
);

export function loadBrowserPreprocessWasmModule(): Effect.Effect<
  BrowserPreprocessWasmModule,
  WorkerRuntimeError
> {
  return Effect.tryPromise({
    try: () =>
      import(
        /* @vite-ignore */
        preprocessWasmModuleUrl.toString()
      ) as Promise<BrowserPreprocessWasmModule>,
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "encoder_preprocessor.load_wasm_module",
        error,
        "failed to load encoder preprocessor wasm module",
      ),
  });
}
