import * as BrowserWorkerRunner from "@effect/platform-browser/BrowserWorkerRunner";
import { Cause, Context, Effect, Layer, Schema, Semaphore } from "effect";

import init, {
  handle_storage_request_json,
  handle_runtime_request_json,
  reset_runtime_state,
} from "../../playwright-harness/pkg/next_plaid_browser_wasm.js";
import { makeWorkerEntrypointLayer } from "../effect/worker-entrypoint.js";
import {
  type WorkerRuntimeError,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import type {
  SearchWorkerRequest,
  SearchWorkerResponse,
} from "../shared/search-contract.js";

const storageRequestTypes = new Set(["install_bundle", "load_stored_bundle"]);
const decodeJsonString = Schema.decodeUnknownEffect(Schema.UnknownFromJsonString);
const encodeJsonString = Schema.encodeUnknownEffect(Schema.UnknownFromJsonString);

function decodeSearchWorkerRequest(
  value: unknown,
): Effect.Effect<SearchWorkerRequest, WorkerRuntimeError> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return workerRuntimeError({
      operation: "search_worker.decode_request",
      message: "search worker request payload must be an object",
      details: value,
    }).asEffect();
  }
  if (typeof (value as { type?: unknown }).type !== "string") {
    return workerRuntimeError({
      operation: "search_worker.decode_request",
      message: "search worker request payload must include a string type",
      details: value,
    }).asEffect();
  }
  return Effect.succeed(value as SearchWorkerRequest);
}

function parseSearchWorkerResponse(
  json: string,
): Effect.Effect<SearchWorkerResponse, WorkerRuntimeError> {
  return decodeJsonString(json).pipe(
    Effect.map((value) => value as SearchWorkerResponse),
    Effect.mapError((error) =>
      workerRuntimeErrorFromUnknown(
        "search_worker.parse_response",
        error,
        "failed to parse search worker response",
      ),
    ),
  );
}

interface SearchWorkerRuntimeApi {
  readonly handleRequest: (
    request: SearchWorkerRequest,
  ) => Effect.Effect<SearchWorkerResponse, WorkerRuntimeError>;
}

class SearchWorkerRuntime
  extends Context.Service<SearchWorkerRuntime, SearchWorkerRuntimeApi>()(
    "next-plaid-browser/SearchWorkerRuntime",
  )
{}

const SearchWorkerRuntimeLive = Layer.effect(
  SearchWorkerRuntime,
  Effect.gen(function*() {
    const requestSemaphore = yield* Semaphore.make(1);
    const ensureRuntime = yield* Effect.cached(
      Effect.tryPromise({
        try: async () => {
          await init();
          reset_runtime_state();
        },
        catch: (error) =>
          workerRuntimeErrorFromUnknown(
            "search_worker.init_runtime",
            error,
            "failed to initialize search runtime",
          ),
      }),
    );

    const handleRequest = Effect.fn("SearchWorkerRuntime.handleRequest")(
      (request: SearchWorkerRequest) =>
        requestSemaphore.withPermit(
          Effect.gen(function*() {
            yield* ensureRuntime;
            const encodedRequest = yield* encodeJsonString(request).pipe(
              Effect.mapError((error) =>
                workerRuntimeErrorFromUnknown(
                  "search_worker.encode_request",
                  error,
                  "failed to encode search worker request",
                ),
              ),
            );
            const responseJson = yield* (
              storageRequestTypes.has(request.type)
                ? Effect.tryPromise({
                    try: () =>
                      handle_storage_request_json(encodedRequest),
                    catch: (error) =>
                      workerRuntimeErrorFromUnknown(
                        "search_worker.storage_request",
                        error,
                        "search worker storage request failed",
                      ),
                  })
                : Effect.try({
                    try: () =>
                      handle_runtime_request_json(encodedRequest),
                    catch: (error) =>
                      workerRuntimeErrorFromUnknown(
                        "search_worker.runtime_request",
                        error,
                        "search worker runtime request failed",
                      ),
                  })
            );

            return yield* parseSearchWorkerResponse(responseJson);
          }),
        ),
    );

    return SearchWorkerRuntime.of({
      handleRequest,
    });
  }),
);

const SearchWorkerLive = makeWorkerEntrypointLayer<
  SearchWorkerRequest,
  SearchWorkerResponse,
  never,
  SearchWorkerRuntime
>({
  workerKind: "search",
  decodeRequest: decodeSearchWorkerRequest,
  handleRequest: (_requestId, request, _emitEvent) =>
    Effect.gen(function*() {
      const runtime = yield* SearchWorkerRuntime;
      return yield* runtime.handleRequest(request);
    }),
}).pipe(
  Layer.provide(SearchWorkerRuntimeLive),
  Layer.provide(BrowserWorkerRunner.layer),
);

Effect.runFork(
  Layer.launch(SearchWorkerLive).pipe(
    Effect.tapCause((cause) =>
      Effect.logError(
        `search worker failed during launch:\n${Cause.pretty(cause)}`,
      ),
    ),
  ),
);
