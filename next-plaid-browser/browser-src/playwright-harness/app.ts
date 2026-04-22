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
  SearchResultsResponseEnvelope,
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
  encoder_build: "tiny-encoder-proof-v2",
  embedding_dim: 4,
  normalized: true,
};
const MUTABLE_CORPUS_ID = "mutable-smoke";

interface RealModelPreset {
  readonly id: string;
  readonly modelId: string;
  readonly modelFile: string;
  readonly embeddingDim: number;
  readonly normalized: boolean;
}

interface RealCorpusDocumentFixture {
  readonly document_id: string;
  readonly semantic_text: string;
  readonly metadata: {
    readonly slug: string;
    readonly title: string;
    readonly source: string;
  };
}

interface RealCorpusQueryFixture {
  readonly id: string;
  readonly text: string;
  readonly expectedSlug: string;
}

interface RealCorpusFixture {
  readonly id: string;
  readonly documents: readonly RealCorpusDocumentFixture[];
  readonly queries: readonly RealCorpusQueryFixture[];
}

const REAL_MODEL_PRESETS: readonly RealModelPreset[] = [
  {
    id: "mxbai-edge-colbert-v0-32m-onnx",
    modelId: "lightonai/mxbai-edge-colbert-v0-32m-onnx",
    modelFile: "model_int8.onnx",
    embeddingDim: 64,
    normalized: true,
  },
  {
    id: "answerai-colbert-small-v1-onnx",
    modelId: "lightonai/answerai-colbert-small-v1-onnx",
    modelFile: "model_int8.onnx",
    embeddingDim: 96,
    normalized: true,
  },
  {
    id: "GTE-ModernColBERT-v1",
    modelId: "lightonai/GTE-ModernColBERT-v1",
    modelFile: "model_int8.onnx",
    embeddingDim: 128,
    normalized: true,
  },
] as const;

const REAL_CORPUS_NEXT_PLAID_DOCS: RealCorpusFixture = {
  id: "next-plaid-docs-v1",
  documents: [
    {
      document_id: "why-multi-vector",
      semantic_text:
        "Standard vector search collapses an entire document into one embedding, which is a lossy summary. Multi-vector retrieval keeps many embeddings per document instead of one. At query time, each query token finds its best match across document tokens with MaxSim. That keeps more detail from names, parameters, docstrings, and control flow than a single vector summary.",
      metadata: {
        slug: "why_multi_vector",
        title: "Why Multi-Vector Retrieval Matters",
        source: "README.md",
      },
    },
    {
      document_id: "api-cpu-quickstart",
      semantic_text:
        "Run NextPlaid API with Docker on CPU using a built-in model. The quick start runs the container, mounts a local data directory, exposes port 8080, and passes the model flag lightonai slash answerai-colbert-small-v1-onnx together with int8 quantization.",
      metadata: {
        slug: "api_cpu_quickstart",
        title: "API CPU Quick Start",
        source: "next-plaid-api/README.md",
      },
    },
    {
      document_id: "api-two-modes",
      semantic_text:
        "The API has two modes depending on whether you pass a model. With a model, callers send text and the server encodes it through ONNX Runtime. Without a model, callers must provide embeddings directly and text encoding endpoints are unavailable. The no-model mode is for custom models and external encoding pipelines.",
      metadata: {
        slug: "api_two_modes",
        title: "API With Model Versus Without Model",
        source: "next-plaid-api/README.md",
      },
    },
    {
      document_id: "ready-to-use-models",
      semantic_text:
        "Ready-to-use models include lightonai slash mxbai-edge-colbert-v0-32m-onnx and lightonai slash answerai-colbert-small-v1-onnx for lightweight text retrieval, and lightonai slash GTE-ModernColBERT-v1 for more accurate text retrieval. LateOn-Code-edge is lightweight for code search, while LateOn-Code is the more accurate code-search model.",
      metadata: {
        slug: "ready_to_use_models",
        title: "Ready To Use Model Guide",
        source: "README.md",
      },
    },
    {
      document_id: "colgrep-overview",
      semantic_text:
        "ColGREP is semantic code search for the terminal and coding agents. Searches combine regex filtering with semantic ranking, stay fully local, and use Tree-sitter structure plus a multi-vector model before ranking with NextPlaid.",
      metadata: {
        slug: "colgrep_overview",
        title: "ColGREP Overview",
        source: "README.md",
      },
    },
  ],
  queries: [
    {
      id: "lightweight-text-model",
      text: "Which model is lightweight for text retrieval?",
      expectedSlug: "ready_to_use_models",
    },
    {
      id: "cpu-docker-model",
      text: "How do I run the API on CPU with a built in model?",
      expectedSlug: "api_cpu_quickstart",
    },
    {
      id: "single-vector-loss",
      text: "Why is multi vector retrieval better than one embedding per document?",
      expectedSlug: "why_multi_vector",
    },
    {
      id: "api-no-model",
      text: "What happens when the API runs without a model?",
      expectedSlug: "api_two_modes",
    },
    {
      id: "what-is-colgrep",
      text: "What is ColGREP used for?",
      expectedSlug: "colgrep_overview",
    },
  ],
} as const;

function setStatus(state: string, value: unknown): void {
  statusNode.dataset.state = state;
  statusNode.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function currentScenario(): string {
  return new URLSearchParams(window.location.search).get("scenario") ?? "wrapper-smoke";
}

function currentRealModelPreset(): RealModelPreset {
  const requested = new URLSearchParams(window.location.search).get("modelPreset") ??
    REAL_MODEL_PRESETS[0].id;
  const preset = REAL_MODEL_PRESETS.find((candidate) => candidate.id === requested);
  if (preset === undefined) {
    throw new Error(`unknown real model preset: ${requested}`);
  }
  return preset;
}

function currentRealCorpusFixture(): RealCorpusFixture {
  const requested = new URLSearchParams(window.location.search).get("corpusPreset") ??
    REAL_CORPUS_NEXT_PLAID_DOCS.id;
  if (requested !== REAL_CORPUS_NEXT_PLAID_DOCS.id) {
    throw new Error(`unknown real corpus preset: ${requested}`);
  }
  return REAL_CORPUS_NEXT_PLAID_DOCS;
}

function huggingFaceResolveUrl(modelId: string, fileName: string): string {
  return `https://huggingface.co/${modelId}/resolve/main/${fileName}`;
}

function realModelEncoderIdentity(preset: RealModelPreset): EncoderIdentity {
  return {
    encoder_id: preset.modelId,
    encoder_build: `${preset.modelFile}@main`,
    embedding_dim: preset.embeddingDim,
    normalized: preset.normalized,
  };
}

function realModelCorpusId(preset: RealModelPreset, corpus: RealCorpusFixture): string {
  return `real-model-${preset.id}-${corpus.id}`;
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
      tokenizerUrl: "../fixtures/encoder-proof/tokenizer.json",
      prefer: "wasm",
    },
  };
}

function realModelEncoderInitRequest(preset: RealModelPreset): EncoderInitRequest {
  return {
    type: "init",
    payload: {
      encoder: realModelEncoderIdentity(preset),
      modelUrl: huggingFaceResolveUrl(preset.modelId, preset.modelFile),
      onnxConfigUrl: huggingFaceResolveUrl(preset.modelId, "onnx_config.json"),
      tokenizerUrl: huggingFaceResolveUrl(preset.modelId, "tokenizer.json"),
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

function realCorpusSearchArgs(corpusId: string, queryText: string) {
  return {
    corpusId,
    queryText,
    request: {
      params: {
        top_k: 3,
        n_ivf_probe: 8,
        n_full_scores: 16,
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

function summarizeRealCorpusSearch(
  query: RealCorpusQueryFixture,
  response: SearchResultsResponseEnvelope,
) {
  const firstResult = response.results[0];
  const returnedMetadata = Array.isArray(firstResult?.metadata) ? firstResult.metadata : [];
  const returnedSlugs = returnedMetadata.map((entry) =>
    entry &&
      typeof entry === "object" &&
      "slug" in entry &&
      typeof entry.slug === "string"
      ? entry.slug
      : null
  );
  const returnedTitles = returnedMetadata.map((entry) =>
    entry &&
      typeof entry === "object" &&
      "title" in entry &&
      typeof entry.title === "string"
      ? entry.title
      : null
  );

  return {
    queryId: query.id,
    queryText: query.text,
    expectedSlug: query.expectedSlug,
    returnedSlugs,
    returnedTitles,
    scores: firstResult?.scores ?? [],
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
        const initialLoadedIndices = yield* SubscriptionRef.get(runtimeService.loadedIndices);
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
            loaded_indices: (yield* SubscriptionRef.get(runtimeService.loadedIndices)).size,
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
            loaded_indices: (yield* SubscriptionRef.get(runtimeService.loadedIndices)).size,
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

async function runRealModelProbe(): Promise<unknown> {
  const modelPreset = currentRealModelPreset();
  const corpus = currentRealCorpusFixture();
  const corpusId = realModelCorpusId(modelPreset, corpus);
  const initRequest = realModelEncoderInitRequest(modelPreset);

  const initialRuntime = makeHarnessRuntime();
  let initialPhase: {
    readonly encoderInitEvents: readonly EncoderInitEvent[];
    readonly encoderCapabilities: unknown;
    readonly registerCorpus: unknown;
    readonly syncCorpus: unknown;
    readonly searches: readonly unknown[];
    readonly mutableCorpusState: unknown;
  };
  try {
    initialPhase = await initialRuntime.runPromise(
      Effect.scoped(
        Effect.gen(function*() {
          const encoderClient = yield* EncoderWorkerClient;
          const runtimeService = yield* BrowserSearchRuntime;
          const encoderEvents: EncoderInitEvent[] = [];

          yield* Stream.runForEach(encoderClient.events, (event) =>
            Effect.sync(() => {
              if (event.stage !== "failed" && event.stage !== "disposed") {
                encoderEvents.push(event);
              }
            }),
          ).pipe(Effect.forkScoped);

          const encoderCapabilities = yield* encoderClient.init(initRequest.payload);
          const registerCorpus = yield* runtimeService.registerCorpus({
            corpusId,
            encoder: {
              encoder_id: encoderCapabilities.encoderId,
              encoder_build: encoderCapabilities.encoderBuild,
              embedding_dim: encoderCapabilities.embeddingDim,
              normalized: encoderCapabilities.normalized,
            },
            ftsTokenizer: "unicode61",
          });
          const syncCorpus = yield* runtimeService.syncCorpus({
            corpusId,
            snapshot: {
              documents: [...corpus.documents],
            },
          });
          const searches = yield* Effect.forEach(
            corpus.queries,
            (query) =>
              runtimeService.searchCorpus(realCorpusSearchArgs(corpusId, query.text)).pipe(
                Effect.map((response) => summarizeRealCorpusSearch(query, response)),
              ),
            { concurrency: 1 },
          );
          const mutableCorpusState = (yield* SubscriptionRef.get(
            runtimeService.mutableCorpora,
          )).get(corpusId) ?? null;

          return {
            encoderInitEvents: [...encoderEvents],
            encoderCapabilities,
            registerCorpus,
            syncCorpus,
            searches,
            mutableCorpusState,
          };
        }),
      ),
    );
  } finally {
    await initialRuntime.dispose();
  }

  const runtime = makeHarnessRuntime();
  try {
    const reloadedPhase = await runtime.runPromise(
      Effect.scoped(
        Effect.gen(function*() {
          const encoderClient = yield* EncoderWorkerClient;
          const runtimeService = yield* BrowserSearchRuntime;
          const encoderEvents: EncoderInitEvent[] = [];

          yield* Stream.runForEach(encoderClient.events, (event) =>
            Effect.sync(() => {
              if (event.stage !== "failed" && event.stage !== "disposed") {
                encoderEvents.push(event);
              }
            }),
          ).pipe(Effect.forkScoped);

          const encoderCapabilities = yield* encoderClient.init(initRequest.payload);
          const searches = yield* Effect.forEach(
            corpus.queries,
            (query) =>
              runtimeService.searchCorpus(realCorpusSearchArgs(corpusId, query.text)).pipe(
                Effect.map((response) => summarizeRealCorpusSearch(query, response)),
              ),
            { concurrency: 1 },
          );
          const syncCorpus = yield* runtimeService.syncCorpus({
            corpusId,
            snapshot: {
              documents: [...corpus.documents],
            },
          });
          const mutableCorpusState = (yield* SubscriptionRef.get(
            runtimeService.mutableCorpora,
          )).get(corpusId) ?? null;

          return {
            encoderInitEvents: [...encoderEvents],
            encoderCapabilities,
            searches,
            syncCorpus,
            mutableCorpusState,
          };
        }),
      ),
    );

    return {
      scenario: "real-model-probe",
      modelPreset: {
        id: modelPreset.id,
        modelId: modelPreset.modelId,
        modelFile: modelPreset.modelFile,
      },
      corpus: {
        id: corpus.id,
        documentCount: corpus.documents.length,
        queryCount: corpus.queries.length,
      },
      initialPhase,
      reloadedPhase,
    };
  } finally {
    await runtime.dispose();
  }
}

async function main(): Promise<void> {
  try {
    const result = currentScenario() === "real-model-probe"
      ? await runRealModelProbe()
      : await runWrapperSmoke();
    window.__NEXT_PLAID_SMOKE_RESULT__ = result;
    setStatus("ok", result);
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    window.__NEXT_PLAID_SMOKE_ERROR__ = message;
    setStatus("error", message);
  }
}

void main();
