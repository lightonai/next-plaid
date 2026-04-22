import { Effect, Stream, SubscriptionRef } from "effect";

import type {
  BundleInstalledResponseEnvelope,
  BundleManifest,
  EncoderIdentity,
  InstallBundleRequestEnvelope,
  LoadIndexRequestEnvelope,
  LoadStoredBundleRequestEnvelope,
  QueryEmbeddingsPayload,
  SearchRequestEnvelope,
} from "../shared/search-contract.js";
import type { EncoderInitEvent, EncoderInitRequest } from "../model-worker/types.js";
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
const MUTABLE_CORPUS_ID = "mutable-smoke";

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

function mutableCorpusRegisterArgs() {
  return {
    corpusId: MUTABLE_CORPUS_ID,
    encoder: PROOF_ENCODER,
    ftsTokenizer: "unicode61" as const,
  };
}

function mutableCorpusSyncArgs() {
  return {
    corpusId: MUTABLE_CORPUS_ID,
    snapshot: {
      documents: [
        {
          document_id: "doc-alpha",
          semantic_text: "alpha launch semantic body",
          metadata: {
            title: "alpha launch memo",
            topic: "edge",
          },
        },
        {
          document_id: "doc-beta",
          semantic_text: "beta report semantic body",
          metadata: {
            title: "beta report summary",
            topic: "metrics",
          },
        },
      ],
    },
  };
}

function mutableCorpusSearchArgs() {
  return {
    corpusId: MUTABLE_CORPUS_ID,
    queryText: "alpha",
    request: {
      params: {
        top_k: 2,
        n_ivf_probe: 2,
        n_full_scores: 2,
        centroid_score_threshold: null,
      },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null,
    },
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

async function runWrapperSmoke(): Promise<unknown> {
  const initialRuntime = makeHarnessRuntime();
  let initialPhase: {
    readonly initialState: unknown;
    readonly initialLoadedIndexCount: number;
    readonly installBundle: BundleInstalledResponseEnvelope;
    readonly registerCorpus: unknown;
    readonly encoderCapabilities: unknown;
    readonly syncCorpus: unknown;
    readonly mutableSearch: unknown;
  };
  try {
    initialPhase = await initialRuntime.runPromise(
      Effect.gen(function*() {
        const searchClient = yield* SearchWorkerClient;
        const encoderClient = yield* EncoderWorkerClient;
        const runtimeService = yield* BrowserSearchRuntime;
        const initialState = yield* SubscriptionRef.get(searchClient.state);
        const initialLoadedIndices = yield* SubscriptionRef.get(searchClient.loadedIndices);
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
        const registerCorpus = yield* runtimeService.registerCorpus(
          mutableCorpusRegisterArgs(),
        );
        const encoderCapabilities = yield* encoderClient.init(encoderInitRequest().payload);
        const syncCorpus = yield* runtimeService.syncCorpus(mutableCorpusSyncArgs());
        const mutableSearch = yield* runtimeService.searchCorpus(
          mutableCorpusSearchArgs(),
        );
        return {
          initialState,
          initialLoadedIndexCount: initialLoadedIndices.size,
          installBundle,
          registerCorpus,
          encoderCapabilities,
          syncCorpus,
          mutableSearch,
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
          const reloadedInitialHealth = {
            loaded_indices: (yield* SubscriptionRef.get(searchClient.loadedIndices)).size,
          };
          const searchState = yield* SubscriptionRef.get(runtimeService.searchState);

          const loadStoredBundle = yield* searchClient.loadStoredBundle(loadStoredBundleRequest());
          const storedSemanticSearch = yield* searchClient.search(storedSemanticSearchRequest());
          const storedKeywordSearch = yield* searchClient.search(storedKeywordSearchRequest());
          const storedHybridSearch = yield* searchClient.search(storedHybridSearchRequest());
          const storedFilteredKeywordSearch = yield* searchClient.search(
            storedFilteredKeywordSearchRequest(),
          );
          const load = yield* searchClient.loadIndex(loadIndexRequest());
          const loadEncodedIndex = yield* searchClient.loadIndex(loadEncodedIndexRequest());
          const semanticSearch = yield* searchClient.search(semanticSearchRequest());
          const keywordSearch = yield* searchClient.search(keywordSearchRequest());
          const hybridSearch = yield* searchClient.search(hybridSearchRequest());
          const filteredSemanticSearch = yield* searchClient.search(
            filteredSemanticSearchRequest(),
          );
          const filteredKeywordSearch = yield* searchClient.search(
            filteredKeywordSearchRequest(),
          );
          const health = {
            loaded_indices: (yield* SubscriptionRef.get(searchClient.loadedIndices)).size,
          };

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
          const encodedQueryValue = yield* encoderClient.encodeQuery({ text: "alpha" });
          const encodedSearch = yield* runtimeService.searchWithEmbeddings(
            encodedSearchRequest(encodedQueryValue.payload),
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
          const mutableReloadedSync = yield* runtimeService.syncCorpus(
            mutableCorpusSyncArgs(),
          );
          const mutableReloadedSearch = yield* runtimeService.searchCorpus(
            mutableCorpusSearchArgs(),
          );
          const mutableCorpusState = (yield* SubscriptionRef.get(
            runtimeService.mutableCorpora,
          )).get(MUTABLE_CORPUS_ID) ?? null;

          return {
            initialHealth: {
              loaded_indices: initialPhase.initialLoadedIndexCount,
            },
            initialState: initialPhase.initialState,
            installBundle: initialPhase.installBundle,
            registerCorpus: initialPhase.registerCorpus,
            initialEncoderCapabilities: initialPhase.encoderCapabilities,
            syncCorpus: initialPhase.syncCorpus,
            mutableSearch: initialPhase.mutableSearch,
            reloadedInitialHealth,
            searchState,
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
            encoderInitEvents: encoderEvents,
            encoderCapabilities,
            encoderInit: {
              type: "encoder_ready" as const,
              state: "ready" as const,
              capabilities: encoderCapabilities,
            },
            encoderHealth: {
              state: encoderState.status,
              capabilities: encoderState.status === "ready"
                ? encoderState.capabilities
                : encoderState.capabilities,
            },
            encoderState,
            encodedQuery: {
              type: "encoded_query" as const,
              encoded: encodedQueryValue,
            },
            encodedSearch,
            runtimeEncodedSearch,
            mutableReloadedSync,
            mutableReloadedSearch,
            mutableCorpusState,
          };
        }),
      ),
    );
    return runtimePhase;
  } finally {
    await runtime.dispose();
  }
}

async function main(): Promise<void> {
  try {
    const result = await runWrapperSmoke();
    window.__NEXT_PLAID_SMOKE_RESULT__ = result;
    setStatus("ok", result);
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    window.__NEXT_PLAID_SMOKE_ERROR__ = message;
    setStatus("error", message);
  }
}

void main();
