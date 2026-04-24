import { Effect, Stream, SubscriptionRef } from "effect";
import { Atom, AtomRegistry } from "effect/unstable/reactivity";

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
  SourceSpan,
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

type HarnessScenario = "interactive-demo" | "wrapper-smoke" | "real-model-probe";

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
  readonly source_span: SourceSpan;
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

function sectionSourceSpan(args: {
  readonly sourceId: string;
  readonly sourceUri: string;
  readonly title: string;
  readonly excerpt: string;
  readonly path: readonly string[];
  readonly anchor: string;
}): SourceSpan {
  return {
    source_id: args.sourceId,
    source_uri: args.sourceUri,
    title: args.title,
    excerpt: args.excerpt,
    locator: {
      type: "section",
      path: [...args.path],
      anchor: args.anchor,
    },
  };
}

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
      source_span: sectionSourceSpan({
        sourceId: "README.md#why_multi_vector",
        sourceUri: "https://github.com/lightonai/next-plaid#why-multi-vector-retrieval-matters",
        title: "Why Multi-Vector Retrieval Matters",
        excerpt:
          "Standard vector search collapses an entire document into one embedding, which is a lossy summary. Multi-vector retrieval keeps many embeddings per document instead of one.",
        path: ["README.md", "Why Multi-Vector Retrieval Matters"],
        anchor: "why_multi_vector",
      }),
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
      source_span: sectionSourceSpan({
        sourceId: "next-plaid-api/README.md#api_cpu_quickstart",
        sourceUri:
          "https://github.com/lightonai/next-plaid/tree/main/next-plaid-api#cpu-quick-start",
        title: "API CPU Quick Start",
        excerpt:
          "Run NextPlaid API with Docker on CPU using a built-in model. The quick start mounts a local data directory, exposes port 8080, and passes the model flag.",
        path: ["next-plaid-api/README.md", "CPU Quick Start"],
        anchor: "api_cpu_quickstart",
      }),
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
      source_span: sectionSourceSpan({
        sourceId: "next-plaid-api/README.md#api_two_modes",
        sourceUri: "https://github.com/lightonai/next-plaid/tree/main/next-plaid-api",
        title: "API With Model Versus Without Model",
        excerpt:
          "The API has two modes depending on whether you pass a model. With a model, callers send text and the server encodes it through ONNX Runtime.",
        path: ["next-plaid-api/README.md", "API Modes"],
        anchor: "api_two_modes",
      }),
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
      source_span: sectionSourceSpan({
        sourceId: "README.md#ready_to_use_models",
        sourceUri: "https://github.com/lightonai/next-plaid#ready-to-use-models",
        title: "Ready To Use Model Guide",
        excerpt:
          "Ready-to-use models include mxbai-edge-colbert and answerai-colbert-small for lightweight text retrieval, and GTE-ModernColBERT for more accurate retrieval.",
        path: ["README.md", "Ready To Use Models"],
        anchor: "ready_to_use_models",
      }),
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
      source_span: sectionSourceSpan({
        sourceId: "README.md#colgrep_overview",
        sourceUri: "https://github.com/lightonai/next-plaid#colgrep",
        title: "ColGREP Overview",
        excerpt:
          "ColGREP is semantic code search for the terminal and coding agents. Searches combine regex filtering with semantic ranking and stay fully local.",
        path: ["README.md", "ColGREP"],
        anchor: "colgrep_overview",
      }),
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

function currentScenario(): HarnessScenario {
  const requested = new URLSearchParams(window.location.search).get("scenario");
  switch (requested) {
    case "wrapper-smoke":
    case "real-model-probe":
    case "interactive-demo":
      return requested;
    default:
      return "interactive-demo";
  }
}

function getRequiredElement<ElementType extends Element>(
  id: string,
  expected: { new (...args: never[]): ElementType },
): ElementType {
  const node = document.getElementById(id);
  if (!(node instanceof expected)) {
    throw new Error(`expected ${id} to be a ${expected.name}`);
  }
  return node;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.stack ?? error.message : String(error);
}

function formatDuration(durationMs: number | null | undefined): string {
  return typeof durationMs === "number" ? `${durationMs.toFixed(1)} ms` : "n/a";
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;")
    .replaceAll("'", "&#39;");
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
    source_spans: [
      sectionSourceSpan({
        sourceId: "demo-alpha.md",
        sourceUri: "https://example.test/demo-alpha",
        title: "alpha launch memo",
        excerpt: "Alpha launch memo excerpt for display-only result context.",
        path: ["demo", "alpha"],
        anchor: "demo-alpha",
      }),
      sectionSourceSpan({
        sourceId: "demo-beta.md",
        sourceUri: "https://example.test/demo-beta",
        title: "beta report summary",
        excerpt: "Beta report summary excerpt for display-only result context.",
        path: ["demo", "beta"],
        anchor: "demo-beta",
      }),
      sectionSourceSpan({
        sourceId: "demo-gamma.md",
        sourceUri: "https://example.test/demo-gamma",
        title: "gamma archive note",
        excerpt: "Gamma archive excerpt for display-only result context.",
        path: ["demo", "gamma"],
        anchor: "demo-gamma",
      }),
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
    source_spans: [
      sectionSourceSpan({
        sourceId: "proof-alpha.md",
        sourceUri: "https://example.test/proof-alpha",
        title: "alpha proof doc",
        excerpt: "Alpha proof document excerpt.",
        path: ["proof", "alpha"],
        anchor: "proof-alpha",
      }),
      sectionSourceSpan({
        sourceId: "proof-beta.md",
        sourceUri: "https://example.test/proof-beta",
        title: "beta proof doc",
        excerpt: "Beta proof document excerpt.",
        path: ["proof", "beta"],
        anchor: "proof-beta",
      }),
      sectionSourceSpan({
        sourceId: "proof-gamma.md",
        sourceUri: "https://example.test/proof-gamma",
        title: "gamma proof doc",
        excerpt: "Gamma proof document excerpt.",
        path: ["proof", "gamma"],
        anchor: "proof-gamma",
      }),
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
          source_span: sectionSourceSpan({
            sourceId: "mutable-alpha.md",
            sourceUri: "https://example.test/mutable-alpha",
            title: "alpha launch memo",
            excerpt: "Alpha mutable corpus source excerpt.",
            path: ["mutable smoke", "alpha"],
            anchor: "mutable-alpha",
          }),
        },
        {
          document_id: "doc-beta",
          semantic_text: "beta report semantic body",
          metadata: {
            title: "beta report summary",
            topic: "metrics",
          },
          source_span: sectionSourceSpan({
            sourceId: "mutable-beta.md",
            sourceUri: "https://example.test/mutable-beta",
            title: "beta report summary",
            excerpt: "Beta mutable corpus source excerpt.",
            path: ["mutable smoke", "beta"],
            anchor: "mutable-beta",
          }),
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
      text_query: ["alpha", "gamma"],
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
  const returnedSourceSpans = Array.isArray(firstResult?.source_spans)
    ? firstResult.source_spans
    : [];
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
  const returnedExcerpts = returnedSourceSpans.map((span) => span?.excerpt ?? null);

  return {
    queryId: query.id,
    queryText: query.text,
    expectedSlug: query.expectedSlug,
    returnedSlugs,
    returnedTitles,
    returnedExcerpts,
    scores: firstResult?.scores ?? [],
  };
}

const demoRuntime = makeHarnessRuntime();

function metadataField(value: unknown, key: string): string | null {
  if (value && typeof value === "object" && key in value) {
    const candidate = value[key as keyof typeof value];
    return typeof candidate === "string" ? candidate : null;
  }
  return null;
}

function setAutomationStatusVisible(visible: boolean): void {
  const section = statusNode.closest("[data-status-section]");
  if (section instanceof HTMLElement) {
    section.hidden = !visible;
  }
}

function setDemoVisible(visible: boolean): void {
  const shell = document.getElementById("demo-shell");
  if (shell instanceof HTMLElement) {
    shell.hidden = !visible;
  }
}

// -----------------------------------------------------------------------------
// Wikipedia corpus fixture loader
// -----------------------------------------------------------------------------

interface WikiDoc {
  readonly document_id: string;
  readonly semantic_text: string;
  readonly metadata: {
    readonly slug: string;
    readonly title: string;
    readonly source: string;
    readonly url: string;
    readonly description: string | null;
  };
  readonly source_span: SourceSpan;
}

interface WikiQuery {
  readonly id: string;
  readonly text: string;
  readonly expectedSlug: string;
}

interface WikiFixture {
  readonly id: string;
  readonly attribution: string;
  readonly documents: readonly WikiDoc[];
  readonly queries: readonly WikiQuery[];
}

async function loadWikiFixture(): Promise<WikiFixture> {
  const response = await fetch("./fixtures/wiki-corpus.json");
  if (!response.ok) {
    throw new Error(
      `failed to load corpus fixture: ${response.status} ${response.statusText}`,
    );
  }
  return (await response.json()) as WikiFixture;
}

// -----------------------------------------------------------------------------
// UI state — single writable atom, rendered via AtomRegistry.subscribe.
// -----------------------------------------------------------------------------

type DemoPhase =
  | { readonly tag: "idle" }
  | { readonly tag: "loading_corpus" }
  | { readonly tag: "initializing" }
  | { readonly tag: "ready" }
  | { readonly tag: "searching"; readonly query: string }
  | { readonly tag: "error"; readonly message: string };

interface InitStage {
  readonly id: string;
  readonly label: string;
  readonly status: "pending" | "running" | "done" | "failed";
  readonly durationMs: number | null;
}

interface EventEntry {
  readonly id: number;
  readonly time: string;
  readonly kind: "info" | "success" | "error" | "encoder" | "sync" | "search";
  readonly label: string;
}

interface ResultRow {
  readonly rank: number;
  readonly documentId: string | number;
  readonly title: string;
  readonly description: string | null;
  readonly excerpt: string;
  readonly url: string | null;
  readonly slug: string;
  readonly score: number | null;
}

interface SearchSnapshot {
  readonly query: string;
  readonly rows: readonly ResultRow[];
  readonly elapsedMs: number | null;
}

type SearchMode = "semantic" | "hybrid";

interface UiState {
  readonly phase: DemoPhase;
  readonly corpus: WikiFixture | null;
  readonly selectedPresetId: string;
  readonly activeCorpusId: string | null;
  readonly initStages: readonly InitStage[];
  readonly events: readonly EventEntry[];
  readonly latestSearch: SearchSnapshot | null;
  readonly runtimeDetail: unknown;
  readonly banner: { readonly tone: "info" | "success" | "error"; readonly text: string } | null;
  readonly searchMode: SearchMode;
  readonly hybridAlpha: number;
}

const INIT_STAGE_DEFS: ReadonlyArray<{ readonly id: string; readonly label: string }> = [
  { id: "fetch_corpus", label: "Load corpus fixture" },
  { id: "session_create_complete", label: "Create encoder session" },
  { id: "warmup_complete", label: "Warm up encoder" },
  { id: "register_corpus", label: "Register corpus" },
  { id: "sync_corpus", label: "Sync documents" },
];

function initialStages(): readonly InitStage[] {
  return INIT_STAGE_DEFS.map((def) => ({
    id: def.id,
    label: def.label,
    status: "pending" as const,
    durationMs: null,
  }));
}

function updateStage(
  stages: readonly InitStage[],
  id: string,
  patch: Partial<InitStage>,
): readonly InitStage[] {
  return stages.map((stage) => (stage.id === id ? { ...stage, ...patch } : stage));
}

const INITIAL_UI_STATE: UiState = {
  phase: { tag: "idle" },
  corpus: null,
  selectedPresetId: REAL_MODEL_PRESETS[0].id,
  activeCorpusId: null,
  initStages: initialStages(),
  events: [],
  latestSearch: null,
  runtimeDetail: null,
  banner: null,
  searchMode: "semantic",
  hybridAlpha: 0.25,
};

let eventSequence = 0;

function pushEvent(
  registry: AtomRegistry.AtomRegistry,
  atom: Atom.Writable<UiState>,
  kind: EventEntry["kind"],
  label: string,
): void {
  const entry: EventEntry = {
    id: ++eventSequence,
    time: new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
    kind,
    label,
  };
  registry.update(atom, (state): UiState => ({
    ...state,
    events: [entry, ...state.events].slice(0, 60),
  }));
}

function mapResponseToRows(response: SearchResultsResponseEnvelope): readonly ResultRow[] {
  const first = response.results[0];
  if (!first) return [];
  const documentIds = Array.isArray(first.document_ids) ? first.document_ids : [];
  const scores = Array.isArray(first.scores) ? first.scores : [];
  const metadata = Array.isArray(first.metadata) ? first.metadata : [];
  const sourceSpans = Array.isArray(first.source_spans) ? first.source_spans : [];

  return documentIds.map((documentId, index) => {
    const entry = metadata[index];
    const sourceSpan = sourceSpans[index] ?? null;
    const title = sourceSpan?.title ?? metadataField(entry, "title") ?? String(documentId);
    const description = metadataField(entry, "description");
    const excerpt = sourceSpan?.excerpt ?? "";
    const url = sourceSpan?.source_uri ?? metadataField(entry, "url");
    const slug = metadataField(entry, "slug") ?? sourceSpan?.source_id ?? `doc-${documentId}`;
    const rawScore = scores[index];
    const score = typeof rawScore === "number" ? rawScore : null;
    return {
      rank: index + 1,
      documentId,
      title,
      description,
      excerpt,
      url,
      slug,
      score,
    };
  });
}

// -----------------------------------------------------------------------------
// Render layer
// -----------------------------------------------------------------------------

interface DemoNodes {
  readonly shell: HTMLElement;
  readonly modelSelect: HTMLSelectElement;
  readonly initializeButton: HTMLButtonElement;
  readonly clearLogButton: HTMLButtonElement;
  readonly statusIndicator: HTMLElement;
  readonly statusText: HTMLElement;
  readonly searchForm: HTMLFormElement;
  readonly searchInput: HTMLInputElement;
  readonly searchSubmit: HTMLButtonElement;
  readonly queryButtons: HTMLElement;
  readonly banner: HTMLElement;
  readonly searchResults: HTMLElement;
  readonly resultsMeta: HTMLElement;
  readonly initStages: HTMLElement;
  readonly initMeta: HTMLElement;
  readonly eventLog: HTMLElement;
  readonly runtimeSummary: HTMLElement;
  readonly documents: HTMLElement;
  readonly corpusCount: HTMLElement;
  readonly corpusDetail: HTMLDetailsElement;
  readonly modeButtons: readonly HTMLButtonElement[];
  readonly alphaGroup: HTMLElement;
  readonly alphaSlider: HTMLInputElement;
  readonly alphaValue: HTMLElement;
}

function getDemoNodes(): DemoNodes {
  const statusIndicator = getRequiredElement("demo-status-indicator", HTMLElement);
  const statusTextNode = statusIndicator.querySelector(".status-text");
  if (!(statusTextNode instanceof HTMLElement)) {
    throw new Error("expected .status-text inside status indicator");
  }
  return {
    shell: getRequiredElement("demo-shell", HTMLElement),
    modelSelect: getRequiredElement("demo-model-select", HTMLSelectElement),
    initializeButton: getRequiredElement("demo-initialize", HTMLButtonElement),
    clearLogButton: getRequiredElement("demo-clear-log", HTMLButtonElement),
    statusIndicator,
    statusText: statusTextNode,
    searchForm: getRequiredElement("demo-search-form", HTMLFormElement),
    searchInput: getRequiredElement("demo-search-input", HTMLInputElement),
    searchSubmit: getRequiredElement("demo-search-submit", HTMLButtonElement),
    queryButtons: getRequiredElement("demo-query-buttons", HTMLElement),
    banner: getRequiredElement("demo-banner", HTMLElement),
    searchResults: getRequiredElement("demo-search-results", HTMLElement),
    resultsMeta: getRequiredElement("demo-results-meta", HTMLElement),
    initStages: getRequiredElement("demo-init-stages", HTMLElement),
    initMeta: getRequiredElement("demo-init-meta", HTMLElement),
    eventLog: getRequiredElement("demo-event-log", HTMLElement),
    runtimeSummary: getRequiredElement("demo-runtime-summary", HTMLElement),
    documents: getRequiredElement("demo-documents", HTMLElement),
    corpusCount: getRequiredElement("demo-corpus-count", HTMLElement),
    corpusDetail: getRequiredElement("demo-corpus-detail", HTMLDetailsElement),
    modeButtons: [
      getRequiredElement("demo-mode-semantic", HTMLButtonElement),
      getRequiredElement("demo-mode-hybrid", HTMLButtonElement),
    ],
    alphaGroup: getRequiredElement("demo-alpha-slider-group", HTMLElement),
    alphaSlider: getRequiredElement("demo-alpha-slider", HTMLInputElement),
    alphaValue: getRequiredElement("demo-alpha-value", HTMLElement),
  };
}

function phaseToStatusAttr(phase: DemoPhase): string {
  switch (phase.tag) {
    case "ready":
      return "ready";
    case "initializing":
    case "loading_corpus":
      return "initializing";
    case "searching":
      return "searching";
    case "error":
      return "error";
    default:
      return "idle";
  }
}

function phaseToStatusText(phase: DemoPhase): string {
  switch (phase.tag) {
    case "idle":
      return "Not initialized";
    case "loading_corpus":
      return "Loading corpus";
    case "initializing":
      return "Initializing encoder";
    case "ready":
      return "Ready";
    case "searching":
      return "Searching";
    case "error":
      return "Error";
  }
}

function renderDemo(state: UiState, nodes: DemoNodes): void {
  nodes.statusIndicator.dataset.status = phaseToStatusAttr(state.phase);
  nodes.statusText.textContent = phaseToStatusText(state.phase);

  const busy =
    state.phase.tag === "loading_corpus" ||
    state.phase.tag === "initializing" ||
    state.phase.tag === "searching";
  nodes.initializeButton.disabled = busy || state.corpus === null;
  nodes.searchInput.disabled = busy || state.activeCorpusId === null;
  nodes.searchSubmit.disabled = busy || state.activeCorpusId === null;
  for (const btn of nodes.queryButtons.querySelectorAll("button")) {
    if (btn instanceof HTMLButtonElement) {
      btn.disabled = busy || state.activeCorpusId === null;
    }
  }

  if (nodes.modelSelect.value !== state.selectedPresetId) {
    nodes.modelSelect.value = state.selectedPresetId;
  }

  for (const btn of nodes.modeButtons) {
    const mode = btn.dataset.mode as SearchMode | undefined;
    const active = mode === state.searchMode;
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  }
  nodes.alphaGroup.hidden = state.searchMode !== "hybrid";
  const alphaStr = state.hybridAlpha.toFixed(2);
  if (nodes.alphaSlider.value !== alphaStr) {
    nodes.alphaSlider.value = alphaStr;
  }
  nodes.alphaValue.textContent = `α ${alphaStr}`;

  if (state.banner === null) {
    nodes.banner.hidden = true;
    nodes.banner.textContent = "";
  } else {
    nodes.banner.hidden = false;
    nodes.banner.dataset.tone = state.banner.tone;
    nodes.banner.textContent = state.banner.text;
  }

  nodes.initStages.innerHTML = state.initStages
    .map(
      (stage) => `
      <li class="stage-row" data-status="${stage.status}">
        <span class="stage-glyph" aria-hidden="true"></span>
        <span class="stage-label">${escapeHtml(stage.label)}</span>
        <span class="stage-duration">${stage.durationMs !== null ? escapeHtml(formatDuration(stage.durationMs)) : ""}</span>
      </li>
    `,
    )
    .join("");
  const doneCount = state.initStages.filter((s) => s.status === "done").length;
  nodes.initMeta.textContent = `${doneCount} / ${state.initStages.length}`;

  if (state.events.length === 0) {
    nodes.eventLog.innerHTML = `<li class="empty-hint">No events yet.</li>`;
  } else {
    nodes.eventLog.innerHTML = state.events
      .map(
        (event) => `
        <li class="event-row" data-kind="${event.kind}">
          <span class="event-time">${escapeHtml(event.time)}</span>
          <span class="event-label">${escapeHtml(event.label)}</span>
        </li>
      `,
      )
      .join("");
  }

  nodes.runtimeSummary.textContent =
    state.runtimeDetail === null
      ? "Run Initialize to populate the runtime snapshot."
      : JSON.stringify(state.runtimeDetail, null, 2);

  if (state.latestSearch === null) {
    const hint =
      state.corpus === null
        ? "Loading corpus fixture…"
        : state.activeCorpusId === null
          ? "Initialize the demo, then type a question or pick a preset."
          : "Type a question or pick a preset query.";
    nodes.resultsMeta.textContent = "—";
    nodes.searchResults.innerHTML = `
      <li class="result-empty">
        <h3>No query yet</h3>
        <p>${escapeHtml(hint)}</p>
      </li>
    `;
  } else {
    const rows = state.latestSearch.rows;
    const elapsed =
      state.latestSearch.elapsedMs !== null
        ? ` · ${formatDuration(state.latestSearch.elapsedMs)}`
        : "";
    nodes.resultsMeta.textContent =
      rows.length === 0
        ? `No matches · "${state.latestSearch.query}"`
        : `${rows.length} match${rows.length === 1 ? "" : "es"}${elapsed}`;
    if (rows.length === 0) {
      nodes.searchResults.innerHTML = `
        <li class="result-empty">
          <h3>No matches for "${escapeHtml(state.latestSearch.query)}"</h3>
          <p>Try different phrasing or one of the preset questions.</p>
        </li>
      `;
    } else {
      nodes.searchResults.innerHTML = rows
        .map(
          (row) => `
          <li class="result-row">
            <span class="result-rank">${row.rank.toString().padStart(2, "0")}</span>
            <div class="result-body">
              <h3 class="result-title">${escapeHtml(row.title)}</h3>
              ${row.description ? `<p class="result-description">${escapeHtml(row.description)}</p>` : ""}
              <p class="result-excerpt">${escapeHtml(row.excerpt)}</p>
              <p class="result-meta">
                ${
                  row.url
                    ? `<a href="${escapeHtml(row.url)}" target="_blank" rel="noreferrer">Read on Simple English Wikipedia</a>`
                    : `<span>${escapeHtml(String(row.documentId))}</span>`
                }
                <span>${escapeHtml(row.slug)}</span>
              </p>
            </div>
            <span class="result-score">${row.score !== null ? row.score.toFixed(3) : "—"}</span>
          </li>
        `,
        )
        .join("");
    }
  }

  if (state.corpus === null) {
    nodes.documents.innerHTML = "";
    nodes.corpusCount.textContent = "";
  } else {
    nodes.corpusCount.textContent = `· ${state.corpus.documents.length} articles`;
    nodes.documents.innerHTML = state.corpus.documents
      .map(
        (doc) => `
        <div class="doc-row">
          <h3 class="doc-title">${escapeHtml(doc.metadata.title)}</h3>
          ${doc.metadata.description ? `<p class="doc-description">${escapeHtml(doc.metadata.description)}</p>` : ""}
          <p class="doc-meta">
            <a href="${escapeHtml(doc.metadata.url)}" target="_blank" rel="noreferrer">Read</a>
          </p>
        </div>
      `,
      )
      .join("");
  }
}

// -----------------------------------------------------------------------------
// Flows — run Effects on the managed runtime, push state into the atom.
// -----------------------------------------------------------------------------

async function runInitializeFlow(
  registry: AtomRegistry.AtomRegistry,
  atom: Atom.Writable<UiState>,
  preset: RealModelPreset,
): Promise<void> {
  const snapshot = registry.get(atom);
  const corpus = snapshot.corpus;
  if (corpus === null) {
    registry.update(atom, (state): UiState => ({
      ...state,
      banner: { tone: "error", text: "Corpus fixture is not loaded yet." },
    }));
    return;
  }

  const corpusId = `wiki-${preset.id}`;
  registry.update(atom, (state): UiState => ({
    ...state,
    phase: { tag: "initializing" },
    initStages: updateStage(
      updateStage(state.initStages, "fetch_corpus", { status: "done" }),
      "session_create_complete",
      { status: "running" },
    ),
    banner: {
      tone: "info",
      text: `Loading ${preset.modelId}. First run downloads model weights — can take 30–120s.`,
    },
  }));
  pushEvent(registry, atom, "info", `Initializing ${preset.modelId}`);

  try {
    const result = await demoRuntime.runPromise(
      Effect.scoped(
        Effect.gen(function*() {
          const encoderClient = yield* EncoderWorkerClient;
          const runtimeService = yield* BrowserSearchRuntime;

          yield* Stream.runForEach(encoderClient.events, (event) =>
            Effect.sync(() => {
              const durationMs =
                "durationMs" in event && typeof event.durationMs === "number"
                  ? event.durationMs
                  : null;
              registry.update(atom, (state): UiState => {
                const known = state.initStages.find((s) => s.id === event.stage);
                if (!known) return state;
                return {
                  ...state,
                  initStages: updateStage(state.initStages, event.stage, {
                    status: "done",
                    durationMs,
                  }),
                };
              });
              pushEvent(
                registry,
                atom,
                "encoder",
                `Encoder · ${event.stage}${durationMs !== null ? ` (${formatDuration(durationMs)})` : ""}`,
              );
            }),
          ).pipe(Effect.forkScoped);

          yield* Stream.runForEach(runtimeService.mutableSyncEvents, (event) =>
            Effect.sync(() => {
              pushEvent(registry, atom, "sync", `Sync · ${event.type}`);
            }),
          ).pipe(Effect.forkScoped);

          const capabilities = yield* encoderClient.init(
            realModelEncoderInitRequest(preset).payload,
          );

          registry.update(atom, (state): UiState => ({
            ...state,
            initStages: updateStage(state.initStages, "register_corpus", {
              status: "running",
            }),
          }));

          const registered = yield* runtimeService.registerCorpus({
            corpusId,
            encoder: {
              encoder_id: capabilities.encoderId,
              encoder_build: capabilities.encoderBuild,
              embedding_dim: capabilities.embeddingDim,
              normalized: capabilities.normalized,
            },
            ftsTokenizer: "unicode61",
          });

          const registeredCreated =
            (registered as { created?: boolean }).created ?? null;
          registry.update(atom, (state): UiState => ({
            ...state,
            initStages: updateStage(
              updateStage(state.initStages, "register_corpus", { status: "done" }),
              "sync_corpus",
              { status: "running" },
            ),
          }));
          pushEvent(
            registry,
            atom,
            "success",
            `Corpus ${registeredCreated === true ? "registered" : registeredCreated === false ? "reopened" : "ready"}`,
          );

          const synced = yield* runtimeService.syncCorpus({
            corpusId,
            snapshot: { documents: [...corpus.documents] },
          });

          const corpusState =
            (yield* SubscriptionRef.get(runtimeService.mutableCorpora)).get(
              corpusId,
            ) ?? null;

          return { capabilities, registered, synced, corpusState };
        }),
      ),
    );

    registry.update(atom, (state): UiState => ({
      ...state,
      phase: { tag: "ready" },
      initStages: updateStage(state.initStages, "sync_corpus", { status: "done" }),
      activeCorpusId: corpusId,
      runtimeDetail: {
        encoder: result.capabilities,
        register: result.registered,
        sync: result.synced,
        corpusState: result.corpusState,
      },
      banner: {
        tone: "success",
        text: `Browser runtime ready · ${corpus.documents.length} articles synced.`,
      },
    }));
    pushEvent(registry, atom, "success", "Ready to search");
  } catch (error) {
    const message = formatError(error);
    registry.update(atom, (state): UiState => ({
      ...state,
      phase: { tag: "error", message },
      initStages: state.initStages.map((stage) =>
        stage.status === "running" ? { ...stage, status: "failed" } : stage,
      ),
      banner: { tone: "error", text: message },
    }));
    pushEvent(registry, atom, "error", `Initialization failed: ${message}`);
  }
}

async function runSearchFlow(
  registry: AtomRegistry.AtomRegistry,
  atom: Atom.Writable<UiState>,
  queryText: string,
): Promise<void> {
  const snapshot = registry.get(atom);
  if (snapshot.corpus === null || snapshot.activeCorpusId === null) {
    registry.update(atom, (state): UiState => ({
      ...state,
      banner: { tone: "error", text: "Initialize the demo before searching." },
    }));
    return;
  }
  if (snapshot.phase.tag === "searching" || snapshot.phase.tag === "initializing") {
    return;
  }

  const corpusId = snapshot.activeCorpusId;
  const mode = snapshot.searchMode;
  const alpha = snapshot.hybridAlpha;

  registry.update(atom, (state): UiState => ({
    ...state,
    phase: { tag: "searching", query: queryText },
    banner: null,
  }));
  const modeLabel = mode === "hybrid" ? `Hybrid α=${alpha.toFixed(2)}` : "Semantic";
  pushEvent(registry, atom, "search", `${modeLabel} · "${queryText}"`);

  const startedAt = performance.now();
  try {
    const response = await demoRuntime.runPromise(
      Effect.gen(function*() {
        const runtimeService = yield* BrowserSearchRuntime;
        return yield* runtimeService.searchCorpus({
          corpusId,
          queryText,
          request: {
            params: {
              top_k: 5,
              n_ivf_probe: 12,
              n_full_scores: 24,
              centroid_score_threshold: null,
            },
            subset: null,
            text_query: mode === "hybrid" ? [queryText] : null,
            alpha: mode === "hybrid" ? alpha : null,
            fusion: mode === "hybrid" ? "relative_score" : null,
            filter_condition: null,
            filter_parameters: null,
          },
        });
      }),
    );
    const elapsedMs = performance.now() - startedAt;
    const rows = mapResponseToRows(response);
    registry.update(atom, (state): UiState => ({
      ...state,
      phase: { tag: "ready" },
      latestSearch: { query: queryText, rows, elapsedMs },
      banner:
        rows.length === 0
          ? { tone: "info", text: `No matches for "${queryText}". Try different phrasing.` }
          : null,
    }));
    pushEvent(
      registry,
      atom,
      rows.length === 0 ? "info" : "success",
      `${rows.length} result${rows.length === 1 ? "" : "s"} in ${formatDuration(elapsedMs)}`,
    );
  } catch (error) {
    const message = formatError(error);
    registry.update(atom, (state): UiState => ({
      ...state,
      phase: { tag: "ready" },
      latestSearch: {
        query: queryText,
        rows: [],
        elapsedMs: performance.now() - startedAt,
      },
      banner: { tone: "error", text: message },
    }));
    pushEvent(registry, atom, "error", `Search failed: ${message}`);
  }
}

// -----------------------------------------------------------------------------
// Bootstrap
// -----------------------------------------------------------------------------

async function runInteractiveDemo(): Promise<void> {
  setAutomationStatusVisible(false);
  setDemoVisible(true);

  const nodes = getDemoNodes();
  const registry = AtomRegistry.make();
  const uiAtom = Atom.make(INITIAL_UI_STATE);
  registry.mount(uiAtom);

  registry.subscribe(uiAtom, (state) => renderDemo(state, nodes), { immediate: true });

  nodes.modelSelect.innerHTML = REAL_MODEL_PRESETS
    .map(
      (preset) => `<option value="${escapeHtml(preset.id)}">${escapeHtml(preset.modelId)}</option>`,
    )
    .join("");
  nodes.modelSelect.value = INITIAL_UI_STATE.selectedPresetId;
  nodes.modelSelect.addEventListener("change", () => {
    registry.update(uiAtom, (state): UiState => ({
      ...state,
      selectedPresetId: nodes.modelSelect.value,
    }));
  });

  // Wire persistent handlers now — they do not depend on the corpus fetch.
  nodes.initializeButton.addEventListener("click", () => {
    const presetId = nodes.modelSelect.value;
    const preset = REAL_MODEL_PRESETS.find((p) => p.id === presetId);
    if (preset === undefined) return;
    void runInitializeFlow(registry, uiAtom, preset);
  });
  nodes.clearLogButton.addEventListener("click", () => {
    registry.update(uiAtom, (state): UiState => ({ ...state, events: [] }));
  });
  nodes.searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    const text = nodes.searchInput.value.trim();
    if (!text) return;
    void runSearchFlow(registry, uiAtom, text);
  });

  for (const btn of nodes.modeButtons) {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode as SearchMode | undefined;
      if (mode !== "semantic" && mode !== "hybrid") return;
      registry.update(uiAtom, (state): UiState => ({ ...state, searchMode: mode }));
    });
  }
  nodes.alphaSlider.addEventListener("input", () => {
    const value = Number.parseFloat(nodes.alphaSlider.value);
    if (!Number.isFinite(value)) return;
    registry.update(uiAtom, (state): UiState => ({
      ...state,
      hybridAlpha: Math.min(1, Math.max(0, value)),
    }));
  });

  registry.update(uiAtom, (state): UiState => ({
    ...state,
    phase: { tag: "loading_corpus" },
    initStages: updateStage(state.initStages, "fetch_corpus", { status: "running" }),
  }));

  try {
    const corpus = await loadWikiFixture();
    registry.update(uiAtom, (state): UiState => ({
      ...state,
      phase: { tag: "idle" },
      corpus,
      initStages: updateStage(state.initStages, "fetch_corpus", { status: "done" }),
      banner: {
        tone: "info",
        text: `Pick an encoder and initialize to sync ${corpus.documents.length} articles into the browser runtime.`,
      },
    }));
    pushEvent(registry, uiAtom, "info", `Corpus loaded · ${corpus.documents.length} docs`);

    nodes.queryButtons.innerHTML = corpus.queries
      .map(
        (query) =>
          `<button class="pill-button" type="button" data-query="${escapeHtml(query.text)}">${escapeHtml(query.text)}</button>`,
      )
      .join("");
    for (const btn of nodes.queryButtons.querySelectorAll("button[data-query]")) {
      btn.addEventListener("click", () => {
        const text = btn.getAttribute("data-query");
        if (text === null) return;
        nodes.searchInput.value = text;
        void runSearchFlow(registry, uiAtom, text);
      });
    }
    nodes.searchInput.value = corpus.queries[0]?.text ?? "";
  } catch (error) {
    const message = formatError(error);
    registry.update(uiAtom, (state): UiState => ({
      ...state,
      phase: { tag: "error", message },
      initStages: updateStage(state.initStages, "fetch_corpus", { status: "failed" }),
      banner: { tone: "error", text: message },
    }));
    pushEvent(registry, uiAtom, "error", `Corpus load failed: ${message}`);
    return;
  }

  window.addEventListener("pagehide", () => {
    void demoRuntime.dispose();
  }, { once: true });
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
  const scenario = currentScenario();
  try {
    if (scenario === "interactive-demo") {
      await runInteractiveDemo();
      return;
    }

    setAutomationStatusVisible(true);
    setDemoVisible(false);

    const result = scenario === "real-model-probe"
      ? await runRealModelProbe()
      : await runWrapperSmoke();
    window.__NEXT_PLAID_SMOKE_RESULT__ = result;
    setStatus("ok", result);
  } catch (error) {
    const message = formatError(error);
    window.__NEXT_PLAID_SMOKE_ERROR__ = message;
    setStatus("error", message);
  }
}

void main();
