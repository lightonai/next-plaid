import { Effect, Stream, SubscriptionRef } from "effect";

import type {
  BundleInstalledResponseEnvelope,
  BundleManifest,
  EncoderIdentity,
  HealthRequestEnvelope,
  HealthResponseEnvelope,
  IndexLoadedResponseEnvelope,
  InstallBundleRequestEnvelope,
  LoadIndexRequestEnvelope,
  LoadStoredBundleRequestEnvelope,
  QueryEmbeddingsPayload,
  SearchResultsResponseEnvelope,
  SearchWorkerRequest,
  SearchWorkerResponse,
  StoredBundleLoadedResponseEnvelope,
  SearchRequestEnvelope,
} from "../shared/search-contract.js";
import type {
  EncodeResponse,
  EncoderDisposeResponse,
  EncoderHealthResponse,
  EncoderInitEvent,
  EncoderInitRequest,
  EncoderInitResponse,
  EncoderWorkerRequest,
} from "../model-worker/types.js";
import type { WorkerResponseEnvelope } from "../shared/worker-envelope.js";
import {
  makeBrowserSearchRuntimeManagedRuntimeFromFactories,
} from "../effect/browser-runtime-app.js";
import { BrowserSearchRuntime } from "../effect/browser-search-runtime.js";
import { permanentClientError } from "../effect/client-errors.js";
import { EncoderWorkerClient } from "../effect/encoder-worker-client.js";
import { SearchWorkerClient } from "../effect/search-worker-client.js";

function makeHarnessRuntime() {
  return makeBrowserSearchRuntimeManagedRuntimeFromFactories({
    searchWorker: () => new Worker("./search-worker.js", { type: "module" }),
    encoderWorker: () => new Worker("./encoder-worker.js", { type: "module" }),
  });
}

declare global {
  interface Window {
    __NEXT_PLAID_SMOKE_RESULT__?: unknown;
    __NEXT_PLAID_SMOKE_ERROR__?: string;
  }
}

const statusNode = (() => {
  const node = document.getElementById("status");
  if (!(node instanceof HTMLElement)) {
    throw new Error("expected status node in smoke harness");
  }
  return node;
})();
const WORKER_REQUEST_TIMEOUT_MS = 15_000;

const DENSE_ENCODER: EncoderIdentity = {
  encoder_id: "demo-smoke-dense",
  encoder_build: "demo-smoke-dense-build-1",
  embedding_dim: 2,
  normalized: true,
};

const STORED_ENCODER: EncoderIdentity = {
  encoder_id: "demo-encoder",
  encoder_build: "demo-build",
  embedding_dim: 4,
  normalized: true,
};

const PROOF_ENCODER: EncoderIdentity = {
  encoder_id: "tiny-encoder-proof",
  encoder_build: "tiny-encoder-proof-v1",
  embedding_dim: 4,
  normalized: true,
};

function setStatus(state: string, value: unknown): void {
  statusNode.dataset.state = state;
  statusNode.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function embeddingPayload(
  encoder: EncoderIdentity,
  embeddings: number[][],
  layout: QueryEmbeddingsPayload["layout"] = "ragged",
): QueryEmbeddingsPayload {
  return {
    embeddings,
    encoder,
    dtype: "f32_le",
    layout,
  };
}

function loadIndexRequest(): LoadIndexRequestEnvelope {
  return {
    type: "load_index",
    name: "demo-smoke",
    encoder: DENSE_ENCODER,
    index: {
      centroids: {
        values: [1.0, 0.0, 0.0, 1.0, 0.7, 0.7],
        rows: 3,
        dim: 2,
      },
      ivf_doc_ids: [0, 2, 1, 2, 0, 1, 2],
      ivf_lengths: [2, 2, 3],
      doc_offsets: [0, 2, 4, 6],
      doc_codes: [0, 2, 1, 2, 2, 2],
      doc_values: [
        1.0, 0.0, 0.7, 0.7,
        0.0, 1.0, 0.7, 0.7,
        0.7, 0.7, 0.7, 0.7,
      ],
    },
    metadata: [
      { title: "alpha launch memo", topic: "edge" },
      { title: "beta report summary", topic: "metrics" },
      { title: "gamma archive note", topic: "history" },
    ],
    nbits: 2,
    fts_tokenizer: "unicode61",
    max_documents: null,
  };
}

function loadEncodedIndexRequest(): LoadIndexRequestEnvelope {
  return {
    type: "load_index",
    name: "encoder-demo",
    encoder: PROOF_ENCODER,
    index: {
      centroids: {
        values: [
          0.0, 1.0, 0.0, 0.0,
          0.0, 0.0, 1.0, 0.0,
          0.0, 0.0, 0.0, 1.0,
        ],
        rows: 3,
        dim: 4,
      },
      ivf_doc_ids: [0, 1, 2],
      ivf_lengths: [1, 1, 1],
      doc_offsets: [0, 1, 2, 3],
      doc_codes: [0, 1, 2],
      doc_values: [
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
      ],
    },
    metadata: [
      { title: "alpha proof doc", topic: "tokens" },
      { title: "beta proof doc", topic: "tokens" },
      { title: "gamma proof doc", topic: "tokens" },
    ],
    nbits: 2,
    fts_tokenizer: "unicode61",
    max_documents: null,
  };
}

async function installStoredBundleRequest(): Promise<InstallBundleRequestEnvelope> {
  const manifest = (await fetch("../fixtures/demo-bundle/manifest.json").then((response) =>
    response.json(),
  )) as BundleManifest;
  const artifacts = await Promise.all(
    manifest.artifacts.map(async (artifact) => {
      const buffer = await fetch(`../fixtures/demo-bundle/${artifact.path}`).then((response) =>
        response.arrayBuffer(),
      );
      const bytes = new Uint8Array(buffer as ArrayBuffer);
      let binary = "";
      for (const byte of bytes) {
        binary += String.fromCharCode(byte);
      }
      return {
        kind: artifact.kind,
        bytes_b64: btoa(binary),
      };
    }),
  );

  return {
    type: "install_bundle",
    manifest: {
      ...manifest,
      index_id: "demo-stored-bundle",
      build_id: "build-demo-stored-001",
    },
    artifacts,
    activate: true,
  };
}

function loadStoredBundleRequest(): LoadStoredBundleRequestEnvelope {
  return {
    type: "load_stored_bundle",
    index_id: "demo-stored-bundle",
    name: "stored-demo",
    fts_tokenizer: "unicode61",
  };
}

function semanticSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: [embeddingPayload(DENSE_ENCODER, [[1.0, 0.0], [0.7, 0.7]])],
      params: { top_k: 2, n_ivf_probe: 2, n_full_scores: 3, centroid_score_threshold: null },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function keywordSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: null,
      params: { top_k: 2, n_ivf_probe: null, n_full_scores: null, centroid_score_threshold: null },
      subset: null,
      text_query: ["alpha"],
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function hybridSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: [embeddingPayload(DENSE_ENCODER, [[0.0, 1.0], [0.7, 0.7]])],
      params: { top_k: 2, n_ivf_probe: 2, n_full_scores: 3, centroid_score_threshold: null },
      subset: null,
      text_query: ["beta"],
      alpha: 0.25,
      fusion: "relative_score",
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function filteredSemanticSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: [embeddingPayload(DENSE_ENCODER, [[1.0, 0.0], [0.7, 0.7]])],
      params: { top_k: 2, n_ivf_probe: 2, n_full_scores: 3, centroid_score_threshold: null },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: "topic = ?",
      filter_parameters: ["metrics"],
    },
  };
}

function filteredKeywordSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: null,
      params: { top_k: 2, n_ivf_probe: null, n_full_scores: null, centroid_score_threshold: null },
      subset: null,
      text_query: ["alpha OR gamma"],
      alpha: null,
      fusion: null,
      filter_condition: "topic IN (?, ?)",
      filter_parameters: ["history", "edge"],
    },
  };
}

function storedKeywordSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: null,
      params: { top_k: 2, n_ivf_probe: null, n_full_scores: null, centroid_score_threshold: null },
      subset: null,
      text_query: ["alpha"],
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function storedSemanticSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: [embeddingPayload(STORED_ENCODER, [[1.0, 0.0, 0.0, 0.0]])],
      params: { top_k: 2, n_ivf_probe: 2, n_full_scores: 2, centroid_score_threshold: null },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function storedHybridSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: [embeddingPayload(STORED_ENCODER, [[0.0, 1.0, 0.0, 0.0]])],
      params: { top_k: 2, n_ivf_probe: 2, n_full_scores: 2, centroid_score_threshold: null },
      subset: null,
      text_query: ["beta"],
      alpha: 0.25,
      fusion: "relative_score",
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function storedFilteredKeywordSearchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: null,
      params: { top_k: 2, n_ivf_probe: null, n_full_scores: null, centroid_score_threshold: null },
      subset: null,
      text_query: ["beta"],
      alpha: null,
      fusion: null,
      filter_condition: "title = ?",
      filter_parameters: ["beta"],
    },
  };
}

function encoderInitRequest(): EncoderInitRequest {
  return {
    type: "init",
    payload: {
      encoder: PROOF_ENCODER,
      modelUrl: "../fixtures/encoder-proof/tiny-encoder.onnx",
      onnxConfigUrl: "../fixtures/encoder-proof/onnx_config.json",
      tokenizerUrl: "../fixtures/encoder-proof/tokenizer-fixture.json",
      prefer: "wasm",
    },
  };
}

function encodedSearchRequest(payload: QueryEmbeddingsPayload): SearchRequestEnvelope {
  return {
    type: "search",
    name: "encoder-demo",
    request: {
      queries: [payload],
      params: { top_k: 2, n_ivf_probe: 3, n_full_scores: 3, centroid_score_threshold: null },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function unwrapSearchResponse<T extends SearchWorkerResponse>(
  response: T,
): Exclude<T, Extract<SearchWorkerResponse, { type: "error" }>> {
  if (response.type === "error") {
    throw new Error(`search worker returned ${response.code}: ${response.message}`);
  }
  return response as Exclude<T, Extract<SearchWorkerResponse, { type: "error" }>>;
}

async function callWorker<TRequest, TResponse, TEvent = never>(
  worker: Worker,
  request: TRequest,
  options?: { onEvent?: (event: TEvent) => void },
): Promise<TResponse> {
  const requestId = crypto.randomUUID();

  return new Promise<TResponse>((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      cleanup();
      reject(
        new Error(
          `Worker request timed out after ${WORKER_REQUEST_TIMEOUT_MS}ms: ${
            (request as { type?: string })?.type ?? "unknown"
          }`,
        ),
      );
    }, WORKER_REQUEST_TIMEOUT_MS);

    const cleanup = (): void => {
      clearTimeout(timeoutId);
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      worker.removeEventListener("messageerror", handleMessageError);
    };

    const handleMessage = (event: MessageEvent<unknown>): void => {
      const frame = event.data;
      if (!Array.isArray(frame) || frame.length < 2 || frame[0] !== 1) {
        return;
      }
      const envelope = frame[1] as WorkerResponseEnvelope<TResponse, TEvent>;
      if (envelope?.requestId !== requestId) {
        return;
      }

      if (envelope.ok && "event" in envelope) {
        options?.onEvent?.(envelope.event);
        return;
      }

      cleanup();
      if (envelope.ok) {
        resolve(envelope.response);
      } else {
        reject(new Error(envelope.error));
      }
    };

    const handleError = (event: ErrorEvent): void => {
      cleanup();
      reject(new Error(`Worker error while handling ${(request as { type?: string })?.type ?? "unknown"}: ${event.message}`));
    };

    const handleMessageError = (): void => {
      cleanup();
      reject(new Error(`Worker messageerror while handling ${(request as { type?: string })?.type ?? "unknown"}`));
    };

    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.addEventListener("messageerror", handleMessageError);
    worker.postMessage([0, { requestId, request }]);
  });
}

async function callSearchWorker<TResponse extends SearchWorkerResponse>(
  worker: Worker,
  request: SearchWorkerRequest,
): Promise<Exclude<TResponse, Extract<SearchWorkerResponse, { type: "error" }>>> {
  const response = await callWorker<SearchWorkerRequest, TResponse>(worker, request);
  return unwrapSearchResponse(response);
}

async function runWrapperSmoke(): Promise<unknown> {
  const initialRuntime = makeHarnessRuntime();
  let initialPhase: unknown;
  try {
    initialPhase = await initialRuntime.runPromise(
      Effect.gen(function*() {
        const searchClient = yield* SearchWorkerClient;
        const initialState = yield* SubscriptionRef.get(searchClient.state);
        const installBundle = yield* searchClient.installBundle(
          yield* Effect.tryPromise({
            try: () => installStoredBundleRequest(),
            catch: (error) =>
              permanentClientError({
                cause: "harness_bundle_request_failed",
                message: error instanceof Error ? error.message : String(error),
                operation: "playwright_harness.wrapper_smoke.install_bundle_request",
                details: error,
              }),
          }),
        );
        return {
          initialState,
          installBundle,
        };
      }),
    );
  } finally {
    await initialRuntime.dispose();
  }

  const runtime = makeHarnessRuntime();
  try {
    const runtimePhase = await runtime.runPromise(
      Effect.scoped(
        Effect.gen(function*() {
          const searchClient = yield* SearchWorkerClient;
          const encoderClient = yield* EncoderWorkerClient;
          const runtimeService = yield* BrowserSearchRuntime;
          const searchState = yield* SubscriptionRef.get(runtimeService.searchState);

          const loadStoredBundle = yield* searchClient.loadStoredBundle(loadStoredBundleRequest());
          const load = yield* searchClient.loadIndex(loadIndexRequest());
          const loadEncodedIndex = yield* searchClient.loadIndex(loadEncodedIndexRequest());
          const semanticSearch = yield* searchClient.search(semanticSearchRequest());
          const hybridSearch = yield* searchClient.search(hybridSearchRequest());

          const encoderEvents: EncoderInitEvent[] = [];
          yield* Stream.runForEach(encoderClient.events, (event) =>
            Effect.sync(() => {
              if (event.stage !== "failed" && event.stage !== "disposed") {
                encoderEvents.push(event);
              }
            }),
          ).pipe(Effect.forkScoped);

          const encoderCapabilities = yield* encoderClient.init(encoderInitRequest().payload);
          const encoderState = yield* SubscriptionRef.get(runtimeService.encoderState);
          const encodedQuery = yield* encoderClient.encode({ text: "alpha" });
          const encodedSearch = yield* runtimeService.searchWithEmbeddings(
            encodedSearchRequest(encodedQuery.payload),
          );
          const runtimeEncodedSearch = yield* runtimeService.encodeAndSearch({
            text: "alpha",
            searchRequest: {
              type: "search",
              name: "encoder-demo",
              request: {
                params: {
                  top_k: 2,
                  n_ivf_probe: 3,
                  n_full_scores: 3,
                  centroid_score_threshold: null,
                },
                subset: null,
                text_query: null,
                alpha: null,
                fusion: null,
                filter_condition: null,
                filter_parameters: null,
              },
            },
          });

          return {
            searchState,
            loadStoredBundle,
            load,
            loadEncodedIndex,
            semanticSearch,
            hybridSearch,
            encoderEvents,
            encoderCapabilities,
            encoderState,
            encodedQuery,
            encodedSearch,
            runtimeEncodedSearch,
          };
        }),
      ),
    );

    return {
      initialPhase,
      runtimePhase,
    };
  } finally {
    await runtime.dispose();
  }
}

async function main(): Promise<void> {
  const worker = new Worker("./search-worker.js", { type: "module" });
  const encoderWorker = new Worker("./encoder-worker.js", { type: "module" });
  let reloadWorker: Worker | null = null;

  try {
    const initialHealth = await callSearchWorker<HealthResponseEnvelope>(
      worker,
      { type: "health" } satisfies HealthRequestEnvelope,
    );
    const installBundle = await callSearchWorker<BundleInstalledResponseEnvelope>(
      worker,
      await installStoredBundleRequest(),
    );
    worker.terminate();

    reloadWorker = new Worker("./search-worker.js", { type: "module" });
    const reloadedInitialHealth = await callSearchWorker<HealthResponseEnvelope>(
      reloadWorker,
      { type: "health" } satisfies HealthRequestEnvelope,
    );
    const loadStoredBundle = await callSearchWorker<StoredBundleLoadedResponseEnvelope>(
      reloadWorker,
      loadStoredBundleRequest(),
    );
    const storedSemanticSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      storedSemanticSearchRequest(),
    );
    const storedKeywordSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      storedKeywordSearchRequest(),
    );
    const storedHybridSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      storedHybridSearchRequest(),
    );
    const storedFilteredKeywordSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      storedFilteredKeywordSearchRequest(),
    );
    const load = await callSearchWorker<IndexLoadedResponseEnvelope>(reloadWorker, loadIndexRequest());
    const loadEncodedIndex = await callSearchWorker<IndexLoadedResponseEnvelope>(
      reloadWorker,
      loadEncodedIndexRequest(),
    );
    const health = await callSearchWorker<HealthResponseEnvelope>(
      reloadWorker,
      { type: "health" } satisfies HealthRequestEnvelope,
    );
    const semanticSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      semanticSearchRequest(),
    );
    const keywordSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      keywordSearchRequest(),
    );
    const hybridSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      hybridSearchRequest(),
    );
    const filteredSemanticSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      filteredSemanticSearchRequest(),
    );
    const filteredKeywordSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      filteredKeywordSearchRequest(),
    );

    const encoderInitEvents: EncoderInitEvent[] = [];
    const encoderInit = await callWorker<EncoderWorkerRequest, EncoderInitResponse, EncoderInitEvent>(
      encoderWorker,
      encoderInitRequest(),
      {
        onEvent: (event) => {
          encoderInitEvents.push(event);
        },
      },
    );
    const encoderHealth = await callWorker<EncoderWorkerRequest, EncoderHealthResponse>(
      encoderWorker,
      { type: "health" },
    );
    const encodedQuery = await callWorker<EncoderWorkerRequest, EncodeResponse>(encoderWorker, {
      type: "encode",
      payload: { text: "alpha" },
    });
    const encodedSearch = await callSearchWorker<SearchResultsResponseEnvelope>(
      reloadWorker,
      encodedSearchRequest(encodedQuery.encoded.payload),
    );
    const disposeEncoder = await callWorker<EncoderWorkerRequest, EncoderDisposeResponse>(
      encoderWorker,
      { type: "dispose" },
    );
    const wrapperSmoke = await runWrapperSmoke();

    const result = {
      initialHealth,
      installBundle,
      reloadedInitialHealth,
      loadStoredBundle,
      storedSemanticSearch,
      storedKeywordSearch,
      storedHybridSearch,
      storedFilteredKeywordSearch,
      load,
      loadEncodedIndex,
      health,
      semanticSearch,
      keywordSearch,
      hybridSearch,
      filteredSemanticSearch,
      filteredKeywordSearch,
      encoderInitEvents,
      encoderInit,
      encoderHealth,
      encodedQuery,
      encodedSearch,
      disposeEncoder,
      wrapperSmoke,
    };
    window.__NEXT_PLAID_SMOKE_RESULT__ = result;
    setStatus("ok", result);
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    window.__NEXT_PLAID_SMOKE_ERROR__ = message;
    setStatus("error", message);
  } finally {
    worker.terminate();
    reloadWorker?.terminate();
    encoderWorker.terminate();
  }
}

void main();
