import { expect, layer } from "@effect/vitest";
import { Context, Effect, Fiber, Layer, Stream, SubscriptionRef } from "effect";

import type {
  EncodedDocument,
  EncodedQuery,
  EncoderCapabilities,
  EncoderCreateInput,
} from "../model-worker/types.js";
import type {
  EncoderClientError,
  SearchClientError,
} from "./client-errors.js";
import {
  makeBrowserSearchRuntimeLayer,
} from "./browser-runtime-app.js";
import {
  permanentClientError,
} from "./client-errors.js";
import { DocumentEmbeddingCacheService } from "./document-embedding-cache-service.js";
import type {
  EncoderStateSnapshot,
  EncoderWorkerClientApi,
} from "./encoder-worker-client.js";
import type {
  EncoderIdentity,
  LoadMutableCorpusRequestEnvelope,
  LoadIndexRequestEnvelope,
  QueryEmbeddingsPayload,
  RegisterMutableCorpusRequestEnvelope,
  SearchRequestEnvelope,
  SearchResultsResponseEnvelope,
  SyncMutableCorpusRequestEnvelope,
} from "../shared/search-contract.js";
import {
  BrowserSearchRuntime,
  type BrowserSearchRuntimeApi,
  type SearchCorpusArgs,
  type SyncCorpusArgs,
} from "./browser-search-runtime.js";
import * as BrowserWorker from "./browser-worker.js";
import { EncoderWorkerClient } from "./encoder-worker-client.js";
import type {
  LoadedSearchIndexMetadata,
  MutableCorpusMetadata,
} from "./search-metadata-catalog.js";
import { SearchMetadataCatalog } from "./search-metadata-catalog.js";
import type {
  SearchWorkerClientApi,
  SearchWorkerState,
} from "./search-worker-client.js";
import { SearchWorkerClient } from "./search-worker-client.js";
import {
  type CapturedRequest,
  type FakeSpawner,
  makeFakeSpawner,
} from "./__tests__/fake-spawner.js";

interface BrowserRuntimeHarnessApi {
  readonly searchFake: FakeSpawner;
  readonly encoderFake: FakeSpawner;
}

class BrowserRuntimeHarness
  extends Context.Service<BrowserRuntimeHarness, BrowserRuntimeHarnessApi>()(
    "next-plaid-browser/tests/BrowserRuntimeHarness",
  )
{}

function makeBrowserRuntimeHarnessLayer(): Layer.Layer<
  BrowserRuntimeHarness | SearchWorkerClient | EncoderWorkerClient | BrowserSearchRuntime,
  SearchClientError | EncoderClientError
> {
  const searchFake = makeFakeSpawner();
  const encoderFake = makeFakeSpawner();
  const appLayer = makeBrowserSearchRuntimeLayer({
    searchWorkerLayer: BrowserWorker.layer((id) => searchFake.spawn(id)),
    encoderWorkerLayer: BrowserWorker.layer((id) => encoderFake.spawn(id)),
  });
  const harnessLayer = Layer.succeed(BrowserRuntimeHarness)(
    BrowserRuntimeHarness.of({
      searchFake,
      encoderFake,
    }),
  );

  return Layer.mergeAll(appLayer, harnessLayer);
}

function makeEncodeFailureFallbackLayer(): Layer.Layer<
  BrowserSearchRuntime,
  never
> {
  const catalogLayer = Layer.effect(SearchMetadataCatalog)(
    Effect.gen(function*() {
      const loadedIndices = yield* SubscriptionRef.make<
        ReadonlyMap<string, LoadedSearchIndexMetadata>
      >(
        new Map<string, LoadedSearchIndexMetadata>([
          [
            "proof-index",
            {
              name: "proof-index",
              source: "load_index" as const,
              summary: indexLoadedResponse().summary,
              encoder: proofEncoder(),
              indexId: null,
              buildId: null,
            },
          ],
        ]),
      );

      return SearchMetadataCatalog.of({
        loadedIndices,
        mutableCorpora: yield* SubscriptionRef.make<
          ReadonlyMap<string, MutableCorpusMetadata>
        >(new Map()),
        rememberLoadedIndex: () => Effect.die("unused rememberLoadedIndex in fallback test"),
        rememberInstalledBundle: () =>
          Effect.die("unused rememberInstalledBundle in fallback test"),
        rememberStoredBundleLoad: () =>
          Effect.die("unused rememberStoredBundleLoad in fallback test"),
        rememberMutableCorpus: () =>
          Effect.die("unused rememberMutableCorpus in fallback test"),
      });
    }),
  );

  const searchLayer = Layer.effect(SearchWorkerClient)(
    Effect.gen(function*() {
      const state = yield* SubscriptionRef.make<SearchWorkerState>({
        status: "ready" as const,
        lastError: null,
      });

      return SearchWorkerClient.of({
        state,
        loadIndex: () => Effect.die("unused loadIndex in fallback test"),
        installBundle: () => Effect.die("unused installBundle in fallback test"),
        loadStoredBundle: () => Effect.die("unused loadStoredBundle in fallback test"),
        registerMutableCorpus: () => Effect.die("unused registerMutableCorpus in fallback test"),
        syncMutableCorpus: () => Effect.die("unused syncMutableCorpus in fallback test"),
        loadMutableCorpus: () => Effect.die("unused loadMutableCorpus in fallback test"),
        search: (request) =>
          Effect.sync(() => {
            expect(request.request.queries).toBeNull();
            expect(request.request.text_query).toEqual(["alpha"]);
            expect(request.request.alpha).toBeNull();
            expect(request.request.fusion).toBeNull();
            return searchResultsResponse();
          }),
      });
    }),
  );

  const encoderLayer = Layer.effect(EncoderWorkerClient)(
    Effect.gen(function*() {
      const state = yield* SubscriptionRef.make<EncoderStateSnapshot>({
        status: "ready" as const,
        capabilities: encoderCapabilities(proofEncoder()),
        lastError: null,
      });

      return EncoderWorkerClient.of({
        state,
        events: Stream.empty,
        init: () => Effect.die("unused init in fallback test"),
        encodeQuery: () =>
          Effect.fail(
            permanentClientError({
              cause: "synthetic_encode_failure",
              message: "encode failed in fallback test",
              operation: "encoder_worker.encode_query",
              details: null,
            }),
          ),
        encodeDocument: () => Effect.die("unused encodeDocument in fallback test"),
      });
    }),
  );

  const documentCacheLayer = Layer.succeed(DocumentEmbeddingCacheService)(
    DocumentEmbeddingCacheService.of({
      get: () => Effect.die("unused document cache in fallback test"),
      clear: () => Effect.void,
    }),
  );

  const appLayer = Layer.mergeAll(
    searchLayer,
    encoderLayer,
    catalogLayer,
    documentCacheLayer,
  );
  return BrowserSearchRuntime.layer().pipe(Layer.provide(appLayer));
}

function proofEncoder(
  overrides: Partial<EncoderIdentity> = {},
): EncoderIdentity {
  return {
    encoder_id: overrides.encoder_id ?? "proof-encoder",
    encoder_build: overrides.encoder_build ?? "proof-build-1",
    embedding_dim: overrides.embedding_dim ?? 4,
    normalized: overrides.normalized ?? true,
  };
}

function alternateEncoder(): EncoderIdentity {
  return proofEncoder({
    encoder_id: "alternate-encoder",
    encoder_build: "alternate-build-1",
  });
}

function encoderInput(
  encoder: EncoderIdentity = proofEncoder(),
): EncoderCreateInput {
  return {
    encoder,
    modelUrl: "/proof/model.onnx",
    onnxConfigUrl: "/proof/onnx_config.json",
    tokenizerUrl: "/proof/tokenizer.json",
    prefer: "wasm",
  };
}

function encoderCapabilities(
  encoder: EncoderIdentity = proofEncoder(),
): EncoderCapabilities {
  return {
    backend: "wasm",
    threaded: false,
    persistentStorage: true,
    encoderId: encoder.encoder_id,
    encoderBuild: encoder.encoder_build,
    embeddingDim: encoder.embedding_dim,
    queryLength: 8,
    documentLength: 16,
    doQueryExpansion: false,
    usesTokenTypeIds: false,
    doLowerCase: false,
    queryOutputLayout: "ragged",
    normalized: encoder.normalized,
  };
}

function encodedQuery(
  encoder: EncoderIdentity = proofEncoder(),
): EncodedQuery {
  return {
    payload: {
      embeddings: [[0.1, 0.2, 0.3, 0.4]],
      encoder,
      dtype: "f32_le",
      layout: "ragged",
    },
    timing: {
      total_ms: 2,
      tokenize_ms: 1,
      inference_ms: 1,
    },
    input_ids: [101, 102],
    attention_mask: [1, 1],
  };
}

function indexLoadRequest(
  encoder: EncoderIdentity = proofEncoder(),
): LoadIndexRequestEnvelope {
  return {
    type: "load_index",
    name: "proof-index",
    encoder,
    index: {},
    metadata: null,
    nbits: 2,
    fts_tokenizer: "unicode61",
    max_documents: null,
  } as unknown as LoadIndexRequestEnvelope;
}

function indexLoadedResponse(dimension = 4) {
  return {
    type: "index_loaded",
    name: "proof-index",
    summary: {
      name: "proof-index",
      num_documents: 3,
      num_embeddings: 6,
      num_partitions: 2,
      dimension,
      nbits: 2,
      avg_doclen: 2,
      has_metadata: true,
      max_documents: null,
    },
  } as const;
}

function initResponse(encoder: EncoderIdentity = proofEncoder()) {
  return {
    type: "encoder_ready",
    state: "ready",
    capabilities: encoderCapabilities(encoder),
  } as const;
}

function encodeResponse(encoder: EncoderIdentity = proofEncoder()) {
  return {
    type: "encoded_query",
    encoded: encodedQuery(encoder),
  } as const;
}

function encodedDocument(): EncodedDocument {
  return {
    payload: {
      values: [0.1, 0.2, 0.3, 0.4],
      rows: 1,
      dim: 4,
    },
    timing: {
      total_ms: 2,
      tokenize_ms: 1,
      inference_ms: 1,
    },
    input_ids: [201],
    attention_mask: [1],
  };
}

function emptyEncodedDocument(): EncodedDocument {
  return {
    payload: {
      values: [],
      rows: 0,
      dim: 0,
    },
    timing: {
      total_ms: 2,
      tokenize_ms: 1,
      inference_ms: 1,
    },
    input_ids: [],
    attention_mask: [],
  };
}

function encodeDocumentResponse() {
  return {
    type: "encoded_document",
    encoded: encodedDocument(),
  } as const;
}

function emptyEncodeDocumentResponse() {
  return {
    type: "encoded_document",
    encoded: emptyEncodedDocument(),
  } as const;
}

function malformedEncodeResponseWithNaN(encoder: EncoderIdentity = proofEncoder()) {
  return {
    type: "encoded_query",
    encoded: {
      payload: {
        embeddings: [[0.1, Number.NaN, 0.3, 0.4]],
        encoder,
        dtype: "f32_le",
        layout: "ragged",
      },
      timing: {
        total_ms: 2,
        tokenize_ms: 1,
        inference_ms: 1,
      },
      input_ids: [101, 102],
      attention_mask: [1, 1],
    },
  } as const;
}

function searchResultsResponse(): SearchResultsResponseEnvelope {
  return {
    type: "search_results",
    results: [
      {
        query_id: 0,
        document_ids: [0, 1],
        scores: [0.9, 0.4],
        metadata: [{ title: "alpha" }, { title: "beta" }],
        source_spans: [null, null],
      },
    ],
    num_queries: 1,
    timing: null,
  };
}

function mutableCorpusSummary(
  documentCount = 2,
  options: {
    readonly hasDenseState?: boolean;
    readonly encoder?: EncoderIdentity;
  } = {},
) {
  return {
    corpus_id: "proof-corpus",
    document_count: documentCount,
    has_keyword_state: true,
    has_dense_state: options.hasDenseState ?? false,
    encoder: options.encoder ?? proofEncoder(),
  } as const;
}

function registerCorpusResponse(
  created = true,
  documentCount = 0,
  options: {
    readonly hasDenseState?: boolean;
    readonly encoder?: EncoderIdentity;
  } = {},
) {
  return {
    type: "mutable_corpus_registered",
    corpus_id: "proof-corpus",
    created,
    summary: mutableCorpusSummary(documentCount, options),
  } as const;
}

function syncCorpusResponse(
  options: {
    readonly documentCount?: number;
    readonly changed?: boolean;
    readonly added?: number;
    readonly updated?: number;
    readonly deleted?: number;
    readonly unchanged?: number;
    readonly hasDenseState?: boolean;
    readonly encoder?: EncoderIdentity;
  } = {},
) {
  const changed = options.changed ?? true;
  return {
    type: "mutable_corpus_synced",
    corpus_id: "proof-corpus",
    summary: mutableCorpusSummary(
      options.documentCount ?? 2,
      {
        ...(options.hasDenseState === undefined ? {} : { hasDenseState: options.hasDenseState }),
        ...(options.encoder === undefined ? {} : { encoder: options.encoder }),
      },
    ),
    sync: {
      changed,
      added: options.added ?? (changed ? 2 : 0),
      updated: options.updated ?? 0,
      deleted: options.deleted ?? 0,
      unchanged: options.unchanged ?? (changed ? 0 : options.documentCount ?? 2),
    },
  } as const;
}

function loadMutableCorpusResponse(
  documentCount = 2,
  options: {
    readonly hasDenseState?: boolean;
    readonly encoder?: EncoderIdentity;
  } = {},
) {
  return {
    type: "mutable_corpus_loaded",
    corpus_id: "proof-corpus",
    summary: mutableCorpusSummary(
      documentCount,
      {
        ...(options.hasDenseState === undefined ? {} : { hasDenseState: options.hasDenseState }),
        ...(options.encoder === undefined ? {} : { encoder: options.encoder }),
      },
    ),
  } as const;
}

function searchRequest(
  payload: QueryEmbeddingsPayload,
): SearchRequestEnvelope {
  return {
    type: "search",
    name: "proof-index",
    request: {
      queries: [payload],
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

function hybridEncodeAndSearchArgs() {
  return {
    text: "alpha",
    searchRequest: {
      type: "search" as const,
      name: "proof-index",
      request: {
        params: {
          top_k: 2,
          n_ivf_probe: 2,
          n_full_scores: 2,
          centroid_score_threshold: null,
        },
        subset: null,
        text_query: ["alpha"],
        alpha: 0.25,
        fusion: "relative_score" as const,
        filter_condition: null,
        filter_parameters: null,
      },
    },
  };
}

function syncCorpusArgs(semanticTextPrefix = ""): SyncCorpusArgs {
  return {
    corpusId: "proof-corpus",
    snapshot: {
      documents: [
        {
          document_id: "doc-alpha",
          semantic_text: `${semanticTextPrefix}alpha semantic body`,
          metadata: {
            title: "alpha",
            topic: "edge",
          },
        },
        {
          document_id: "doc-beta",
          semantic_text: `${semanticTextPrefix}beta semantic body`,
          metadata: {
            title: "beta",
            topic: "metrics",
          },
        },
      ],
    },
  };
}

function syncCorpusArgsWithEmbeddings(): SyncCorpusArgs {
  return {
    corpusId: "proof-corpus",
    snapshot: {
      documents: syncCorpusArgs().snapshot.documents.map((document) => ({
        ...document,
        semantic_embeddings: encodedDocument().payload,
      })),
    },
  };
}

function syncCorpusArgsWithRepeatedText(semanticText = "shared semantic body"): SyncCorpusArgs {
  return {
    corpusId: "proof-corpus",
    snapshot: {
      documents: [
        {
          document_id: "doc-alpha",
          semantic_text: semanticText,
          metadata: {
            title: "alpha",
            topic: "edge",
          },
        },
        {
          document_id: "doc-beta",
          semantic_text: semanticText,
          metadata: {
            title: "beta",
            topic: "metrics",
          },
        },
      ],
    },
  };
}

function searchCorpusArgs(): SearchCorpusArgs {
  return {
    corpusId: "proof-corpus",
    request: {
      params: {
        top_k: 2,
        n_ivf_probe: null,
        n_full_scores: null,
        centroid_score_threshold: null,
      },
      subset: null,
      text_query: ["alpha"],
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function denseSearchCorpusArgs(): SearchCorpusArgs {
  return {
    corpusId: "proof-corpus",
    queryText: "alpha dense",
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

function hybridSearchCorpusArgs(): SearchCorpusArgs {
  return {
    corpusId: "proof-corpus",
    queryText: "alpha hybrid",
    request: {
      params: {
        top_k: 2,
        n_ivf_probe: 2,
        n_full_scores: 2,
        centroid_score_threshold: null,
      },
      subset: null,
      text_query: ["alpha hybrid"],
      alpha: 0.25,
      fusion: "relative_score",
      filter_condition: null,
      filter_parameters: null,
    },
  };
}

function firstCapturedRequest<TRequest = unknown>(
  fake: FakeSpawner,
): CapturedRequest<TRequest> {
  const request = fake.capturedRequests<TRequest>()[0];
  expect(request).toBeDefined();
  if (request === undefined) {
    throw new Error("expected a captured worker request");
  }
  return request;
}

function waitForWorkerStart(fake: FakeSpawner): Effect.Effect<void> {
  return Effect.gen(function*() {
    while (!fake.isStarted()) {
      yield* Effect.yieldNow;
    }
  });
}

function waitForCapturedRequestCount(
  fake: FakeSpawner,
  expectedMinimum: number,
): Effect.Effect<void> {
  return Effect.gen(function*() {
    while (fake.capturedRequests().length < expectedMinimum) {
      yield* Effect.yieldNow;
    }
  });
}

function waitForCapturedRequestType<
  TRequest extends { readonly type: string },
>(
  fake: FakeSpawner,
  requestType: TRequest["type"],
): Effect.Effect<CapturedRequest<TRequest>> {
  return Effect.gen(function*() {
    while (true) {
      const request = fake.capturedRequests<TRequest>().find((captured) =>
        captured.request.type === requestType
      );
      if (request !== undefined) {
        return request;
      }
      yield* Effect.yieldNow;
    }
  });
}

function replySuccess(
  fake: FakeSpawner,
  requestId: string,
  response: unknown,
): Effect.Effect<void> {
  return Effect.sync(() => {
    fake.dispatchEnvelope({
      requestId,
      ok: true,
      response,
    });
  });
}

function loadIndexIntoRuntime(
  searchClient: SearchWorkerClientApi,
  fake: FakeSpawner,
  encoder: EncoderIdentity = proofEncoder(),
): Effect.Effect<void, SearchClientError> {
  return Effect.gen(function*() {
    const loadFiber = yield* searchClient.loadIndex(indexLoadRequest(encoder)).pipe(
      Effect.forkChild({ startImmediately: true }),
    );
    yield* Effect.yieldNow;

    const request = firstCapturedRequest<LoadIndexRequestEnvelope>(fake);
    yield* replySuccess(fake, request.requestId, indexLoadedResponse(encoder.embedding_dim));
    const result = yield* Fiber.join(loadFiber);
    expect(result.type).toBe("index_loaded");

    fake.clearOutbound();
  });
}

function initEncoder(
  encoderClient: EncoderWorkerClientApi,
  fake: FakeSpawner,
  encoder: EncoderIdentity = proofEncoder(),
): Effect.Effect<void, EncoderClientError> {
  return Effect.gen(function*() {
    const initFiber = yield* encoderClient.init(encoderInput(encoder)).pipe(
      Effect.forkChild({ startImmediately: true }),
    );
    yield* Effect.yieldNow;

    const snapshot = yield* SubscriptionRef.get(encoderClient.state);
    if (snapshot.status !== "ready") {
      yield* waitForCapturedRequestCount(fake, 1);
      const request = firstCapturedRequest<{ readonly type: "init" }>(fake);
      expect(request.request.type).toBe("init");
      yield* replySuccess(fake, request.requestId, initResponse(encoder));
    }

    const capabilities = yield* Fiber.join(initFiber);
    expect(capabilities).toEqual(encoderCapabilities(encoder));

    fake.clearOutbound();
  });
}

function persistedMutableCorpusMetadata(
  documentCount = 2,
  options: {
    readonly hasDenseState?: boolean;
    readonly encoder?: EncoderIdentity;
  } = {},
): MutableCorpusMetadata {
  return {
    corpusId: "proof-corpus",
    summary: mutableCorpusSummary(documentCount, options),
    loaded: false,
  };
}

function registerMutableCorpus(
  runtime: BrowserSearchRuntimeApi,
  fake: FakeSpawner,
  options: {
    readonly encoder?: EncoderIdentity;
    readonly hasDenseState?: boolean;
  } = {},
): Effect.Effect<void, SearchClientError | EncoderClientError> {
  return Effect.gen(function*() {
    fake.clearOutbound();

    const registerFiber = yield* runtime.registerCorpus({
      corpusId: "proof-corpus",
      encoder: options.encoder ?? proofEncoder(),
    }).pipe(Effect.forkChild({ startImmediately: true }));
    yield* Effect.yieldNow;

    const request = firstCapturedRequest<RegisterMutableCorpusRequestEnvelope>(fake);
    expect(request.request.type).toBe("register_mutable_corpus");
    yield* replySuccess(
      fake,
      request.requestId,
      registerCorpusResponse(true, 0, {
        ...(options.hasDenseState === undefined ? {} : { hasDenseState: options.hasDenseState }),
        ...(options.encoder === undefined ? {} : { encoder: options.encoder }),
      }),
    );
    const result = yield* Fiber.join(registerFiber);
    expect(result.type).toBe("mutable_corpus_registered");
    fake.clearOutbound();
  });
}

layer(makeBrowserRuntimeHarnessLayer())("BrowserSearchRuntime encode and search flow", (it) => {
  it.effect("encodes with the active encoder and forwards the payload into search", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake);
      yield* initEncoder(encoderClient, harness.encoderFake);

      const searchFiber = yield* runtime.encodeAndSearch({
        text: "alpha",
        searchRequest: {
          type: "search",
          name: "proof-index",
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
        },
      }).pipe(Effect.forkChild({ startImmediately: true }));
      yield* Effect.yieldNow;

      const encodeRequest =
        firstCapturedRequest<{ readonly type: "encode_query" }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode_query");
      yield* replySuccess(harness.encoderFake, encodeRequest.requestId, encodeResponse());
      yield* Effect.yieldNow;

      const searchRequestEnvelope = firstCapturedRequest<SearchRequestEnvelope>(harness.searchFake);
      expect(searchRequestEnvelope.request.type).toBe("search");
      expect(searchRequestEnvelope.request.request.queries).toEqual([encodedQuery().payload]);
      yield* replySuccess(
        harness.searchFake,
        searchRequestEnvelope.requestId,
        searchResultsResponse(),
      );

      const result = yield* Fiber.join(searchFiber);
      expect(result.type).toBe("search_results");
      expect(result.results[0]?.document_ids).toEqual([0, 1]);
    }),
  );
});

layer(makeBrowserRuntimeHarnessLayer())("BrowserSearchRuntime mutable corpus API", (it) => {
  it.effect("registers mutable corpora through the browser-owned runtime API", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();
      harness.searchFake.clearOutbound();
      harness.encoderFake.clearOutbound();

      const registerFiber = yield* runtime.registerCorpus({
        corpusId: "proof-corpus",
        encoder: proofEncoder(),
      }).pipe(Effect.forkChild({ startImmediately: true }));
      yield* Effect.yieldNow;

      const request = firstCapturedRequest<RegisterMutableCorpusRequestEnvelope>(harness.searchFake);
      expect(request.request.type).toBe("register_mutable_corpus");
      expect(request.request.corpus_id).toBe("proof-corpus");
      expect(request.request.fts_tokenizer).toBe("unicode61");

      yield* replySuccess(harness.searchFake, request.requestId, registerCorpusResponse());

      const result = yield* Fiber.join(registerFiber);
      expect(result.type).toBe("mutable_corpus_registered");
      expect(result.created).toBe(true);

      const mutableCorpora = yield* SubscriptionRef.get(runtime.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(0),
        loaded: false,
      });
    }),
  );

  it.effect("encodes missing mutable document embeddings before syncing through storage", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* registerMutableCorpus(runtime, harness.searchFake);
      yield* initEncoder(encoderClient, harness.encoderFake);

      const syncFiber = yield* runtime.syncCorpus(syncCorpusArgs()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 1);
      const firstEncodeRequest = firstCapturedRequest<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>(harness.encoderFake);
      expect(firstEncodeRequest.request.type).toBe("encode_document");
      expect(firstEncodeRequest.request.payload.text).toBe("alpha semantic body");
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
      yield* replySuccess(
        harness.encoderFake,
        firstEncodeRequest.requestId,
        encodeDocumentResponse(),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 2);
      const secondEncodeRequest = harness.encoderFake.capturedRequests<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>()[1];
      expect(secondEncodeRequest).toBeDefined();
      if (secondEncodeRequest === undefined) {
        throw new Error("expected second captured document encode request");
      }
      expect(secondEncodeRequest.request.type).toBe("encode_document");
      expect(secondEncodeRequest.request.payload.text).toBe("beta semantic body");
      yield* replySuccess(
        harness.encoderFake,
        secondEncodeRequest.requestId,
        encodeDocumentResponse(),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.searchFake, 1);
      const syncRequest = firstCapturedRequest<SyncMutableCorpusRequestEnvelope>(harness.searchFake);
      expect(syncRequest.request.type).toBe("sync_mutable_corpus");
      expect(syncRequest.request.snapshot.documents).toEqual([
        {
          ...syncCorpusArgs().snapshot.documents[0],
          semantic_embeddings: encodedDocument().payload,
        },
        {
          ...syncCorpusArgs().snapshot.documents[1],
          semantic_embeddings: encodedDocument().payload,
        },
      ]);
      yield* replySuccess(
        harness.searchFake,
        syncRequest.requestId,
        syncCorpusResponse({ hasDenseState: true }),
      );

      const result = yield* Fiber.join(syncFiber);
      expect(result.type).toBe("mutable_corpus_synced");
      expect(result.summary.has_dense_state).toBe(true);

      const mutableCorpora = yield* SubscriptionRef.get(runtime.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(2, { hasDenseState: true }),
        loaded: true,
      });
    }),
  );

  it.effect("reuses one document encode for repeated semantic text within the same sync", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* registerMutableCorpus(runtime, harness.searchFake);
      yield* initEncoder(encoderClient, harness.encoderFake);

      const syncFiber = yield* runtime.syncCorpus(syncCorpusArgsWithRepeatedText()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 1);
      const encodeRequest = firstCapturedRequest<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode_document");
      expect(encodeRequest.request.payload.text).toBe("shared semantic body");
      yield* replySuccess(
        harness.encoderFake,
        encodeRequest.requestId,
        encodeDocumentResponse(),
      );
      yield* Effect.yieldNow;

      expect(harness.encoderFake.capturedRequests()).toHaveLength(1);

      yield* waitForCapturedRequestCount(harness.searchFake, 1);
      const syncRequest = firstCapturedRequest<SyncMutableCorpusRequestEnvelope>(harness.searchFake);
      expect(syncRequest.request.type).toBe("sync_mutable_corpus");
      expect(syncRequest.request.snapshot.documents).toEqual([
        {
          ...syncCorpusArgsWithRepeatedText().snapshot.documents[0],
          semantic_embeddings: encodedDocument().payload,
        },
        {
          ...syncCorpusArgsWithRepeatedText().snapshot.documents[1],
          semantic_embeddings: encodedDocument().payload,
        },
      ]);
      yield* replySuccess(
        harness.searchFake,
        syncRequest.requestId,
        syncCorpusResponse({ hasDenseState: true }),
      );

      const result = yield* Fiber.join(syncFiber);
      expect(result.type).toBe("mutable_corpus_synced");
      expect(result.summary.has_dense_state).toBe(true);
    }),
  );

  it.effect("reuses cached document embeddings across later syncs in the same runtime", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;
      const args = syncCorpusArgsWithRepeatedText("later shared semantic body");

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* registerMutableCorpus(runtime, harness.searchFake);
      yield* initEncoder(encoderClient, harness.encoderFake);

      const firstSyncFiber = yield* runtime.syncCorpus(args).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 1);
      const firstEncodeRequest = firstCapturedRequest<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>(harness.encoderFake);
      expect(firstEncodeRequest.request.payload.text).toBe(
        "later shared semantic body",
      );
      yield* replySuccess(
        harness.encoderFake,
        firstEncodeRequest.requestId,
        encodeDocumentResponse(),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.searchFake, 1);
      const firstSyncRequest = firstCapturedRequest<SyncMutableCorpusRequestEnvelope>(
        harness.searchFake,
      );
      yield* replySuccess(
        harness.searchFake,
        firstSyncRequest.requestId,
        syncCorpusResponse({ hasDenseState: true }),
      );
      yield* Fiber.join(firstSyncFiber);

      harness.searchFake.clearOutbound();
      harness.encoderFake.clearOutbound();

      const secondSyncFiber = yield* runtime.syncCorpus(args).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.searchFake, 1);
      expect(harness.encoderFake.capturedRequests()).toHaveLength(0);

      const secondSyncRequest = firstCapturedRequest<SyncMutableCorpusRequestEnvelope>(
        harness.searchFake,
      );
      expect(secondSyncRequest.request.snapshot.documents).toEqual([
        {
          ...args.snapshot.documents[0],
          semantic_embeddings: encodedDocument().payload,
        },
        {
          ...args.snapshot.documents[1],
          semantic_embeddings: encodedDocument().payload,
        },
      ]);
      yield* replySuccess(
        harness.searchFake,
        secondSyncRequest.requestId,
        syncCorpusResponse({
          changed: false,
          documentCount: 2,
          hasDenseState: true,
        }),
      );

      const secondResult = yield* Fiber.join(secondSyncFiber);
      expect(secondResult.type).toBe("mutable_corpus_synced");
      expect(secondResult.sync.changed).toBe(false);
    }),
  );

  it.effect("reloads mutable corpus metadata from storage before first sync after wrapper state resets", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;
      const args = syncCorpusArgs("reload ");

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* registerMutableCorpus(runtime, harness.searchFake);
      yield* initEncoder(encoderClient, harness.encoderFake);
      yield* SubscriptionRef.set(runtime.mutableCorpora, new Map());
      harness.searchFake.clearOutbound();
      harness.encoderFake.clearOutbound();

      const syncFiber = yield* runtime.syncCorpus(args).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const loadRequest = yield* waitForCapturedRequestType<LoadMutableCorpusRequestEnvelope>(
        harness.searchFake,
        "load_mutable_corpus",
      );
      expect(loadRequest.request.corpus_id).toBe("proof-corpus");
      yield* replySuccess(
        harness.searchFake,
        loadRequest.requestId,
        loadMutableCorpusResponse(0),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 1);
      const firstEncodeRequest = firstCapturedRequest<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>(harness.encoderFake);
      expect(firstEncodeRequest.request.type).toBe("encode_document");
      expect(firstEncodeRequest.request.payload.text).toBe(
        "reload alpha semantic body",
      );
      yield* replySuccess(
        harness.encoderFake,
        firstEncodeRequest.requestId,
        encodeDocumentResponse(),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 2);
      const secondEncodeRequest = harness.encoderFake.capturedRequests<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>()[1];
      expect(secondEncodeRequest).toBeDefined();
      if (secondEncodeRequest === undefined) {
        throw new Error("expected second captured document encode request");
      }
      expect(secondEncodeRequest.request.payload.text).toBe(
        "reload beta semantic body",
      );
      yield* replySuccess(
        harness.encoderFake,
        secondEncodeRequest.requestId,
        encodeDocumentResponse(),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.searchFake, 2);
      const syncRequest = harness.searchFake.capturedRequests<SyncMutableCorpusRequestEnvelope>()[1];
      expect(syncRequest).toBeDefined();
      if (syncRequest === undefined) {
        throw new Error("expected sync request after metadata reload");
      }
      expect(syncRequest.request.type).toBe("sync_mutable_corpus");
      yield* replySuccess(
        harness.searchFake,
        syncRequest.requestId,
        syncCorpusResponse({ hasDenseState: true }),
      );

      const result = yield* Fiber.join(syncFiber);
      expect(result.type).toBe("mutable_corpus_synced");
      expect(result.summary.has_dense_state).toBe(true);

      const mutableCorpora = yield* SubscriptionRef.get(runtime.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(2, { hasDenseState: true }),
        loaded: true,
      });
    }),
  );

  it.effect("fails sync with a document-specific error when encoded document rows are empty", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;
      const args = syncCorpusArgs("invalid ");

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* registerMutableCorpus(runtime, harness.searchFake);
      yield* initEncoder(encoderClient, harness.encoderFake);

      const syncFiber = yield* runtime.syncCorpus(args).pipe(
        Effect.result,
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 1);
      const encodeRequest = firstCapturedRequest<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode_document");
      expect(encodeRequest.request.payload.text).toBe(
        "invalid alpha semantic body",
      );
      yield* replySuccess(
        harness.encoderFake,
        encodeRequest.requestId,
        emptyEncodeDocumentResponse(),
      );

      const syncResult = yield* Fiber.join(syncFiber);
      expect(syncResult._tag).toBe("Failure");
      if (syncResult._tag !== "Failure") {
        throw new Error("expected sync to fail for empty encoded document rows");
      }
      expect(syncResult.failure.cause).toBe("invalid_document_embeddings");
      expect(syncResult.failure.message).toContain("doc-alpha");
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
      expect(harness.encoderFake.capturedRequests()).toHaveLength(1);
    }),
  );

  it.effect("lazy-loads persisted mutable corpora on first mutable search", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();
      yield* SubscriptionRef.set(
        runtime.mutableCorpora,
        new Map([["proof-corpus", persistedMutableCorpusMetadata()]]),
      );
      harness.searchFake.clearOutbound();
      harness.encoderFake.clearOutbound();

      const searchFiber = yield* runtime.searchCorpus(searchCorpusArgs()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const loadRequest = yield* waitForCapturedRequestType<LoadMutableCorpusRequestEnvelope>(
        harness.searchFake,
        "load_mutable_corpus",
      );
      expect(loadRequest.request.type).toBe("load_mutable_corpus");
      expect(loadRequest.request.corpus_id).toBe("proof-corpus");
      yield* replySuccess(
        harness.searchFake,
        loadRequest.requestId,
        loadMutableCorpusResponse(),
      );
      yield* Effect.yieldNow;

      const searchRequestEnvelope = yield* waitForCapturedRequestType<SearchRequestEnvelope>(
        harness.searchFake,
        "search",
      );
      expect(searchRequestEnvelope.request.type).toBe("search");
      expect(searchRequestEnvelope.request.name).toBe("proof-corpus");
      expect(searchRequestEnvelope.request.request.queries).toBeNull();
      expect(searchRequestEnvelope.request.request.text_query).toEqual(["alpha"]);
      expect(searchRequestEnvelope.request.request.alpha).toBeNull();
      expect(searchRequestEnvelope.request.request.fusion).toBeNull();

      yield* replySuccess(
        harness.searchFake,
        searchRequestEnvelope.requestId,
        searchResultsResponse(),
      );

      const result = yield* Fiber.join(searchFiber);
      expect(result.type).toBe("search_results");

      const mutableCorpora = yield* SubscriptionRef.get(runtime.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(),
        loaded: true,
      });
    }),
  );

  it.effect("encodes query text for dense mutable-corpus search after lazy reopen", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* initEncoder(encoderClient, harness.encoderFake);
      yield* SubscriptionRef.set(
        runtime.mutableCorpora,
        new Map([["proof-corpus", persistedMutableCorpusMetadata(2, { hasDenseState: true })]]),
      );
      harness.searchFake.clearOutbound();

      const searchFiber = yield* runtime.searchCorpus(denseSearchCorpusArgs()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const loadRequest = firstCapturedRequest<LoadMutableCorpusRequestEnvelope>(harness.searchFake);
      expect(loadRequest.request.type).toBe("load_mutable_corpus");
      yield* replySuccess(
        harness.searchFake,
        loadRequest.requestId,
        loadMutableCorpusResponse(2, { hasDenseState: true }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 1);
      const encodeRequest = firstCapturedRequest<{
        readonly type: "encode_query";
        readonly payload: { readonly text: string };
      }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode_query");
      expect(encodeRequest.request.payload.text).toBe("alpha dense");
      yield* replySuccess(harness.encoderFake, encodeRequest.requestId, encodeResponse());
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.searchFake, 2);
      const searchRequestEnvelope =
        harness.searchFake.capturedRequests<SearchRequestEnvelope>()[1];
      expect(searchRequestEnvelope).toBeDefined();
      if (searchRequestEnvelope === undefined) {
        throw new Error("expected mutable dense search request after query encoding");
      }
      expect(searchRequestEnvelope.request.type).toBe("search");
      expect(searchRequestEnvelope.request.request.queries).toEqual([encodedQuery().payload]);
      expect(searchRequestEnvelope.request.request.text_query).toBeNull();
      yield* replySuccess(
        harness.searchFake,
        searchRequestEnvelope.requestId,
        searchResultsResponse(),
      );

      const result = yield* Fiber.join(searchFiber);
      expect(result.type).toBe("search_results");
      expect(result.results[0]?.document_ids).toEqual([0, 1]);
    }),
  );

  it.effect("preserves keyword fusion fields for hybrid mutable-corpus search", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* initEncoder(encoderClient, harness.encoderFake);
      yield* SubscriptionRef.set(
        runtime.mutableCorpora,
        new Map([["proof-corpus", persistedMutableCorpusMetadata(2, { hasDenseState: true })]]),
      );
      harness.searchFake.clearOutbound();

      const searchFiber = yield* runtime.searchCorpus(hybridSearchCorpusArgs()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const loadRequest = firstCapturedRequest<LoadMutableCorpusRequestEnvelope>(harness.searchFake);
      yield* replySuccess(
        harness.searchFake,
        loadRequest.requestId,
        loadMutableCorpusResponse(2, { hasDenseState: true }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.encoderFake, 1);
      const encodeRequest = firstCapturedRequest<{
        readonly type: "encode_query";
        readonly payload: { readonly text: string };
      }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode_query");
      expect(encodeRequest.request.payload.text).toBe("alpha hybrid");
      yield* replySuccess(harness.encoderFake, encodeRequest.requestId, encodeResponse());
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.searchFake, 2);
      const searchRequestEnvelope =
        harness.searchFake.capturedRequests<SearchRequestEnvelope>()[1];
      expect(searchRequestEnvelope).toBeDefined();
      if (searchRequestEnvelope === undefined) {
        throw new Error("expected mutable hybrid search request after query encoding");
      }
      expect(searchRequestEnvelope.request.request.queries).toEqual([encodedQuery().payload]);
      expect(searchRequestEnvelope.request.request.text_query).toEqual(["alpha hybrid"]);
      expect(searchRequestEnvelope.request.request.alpha).toBe(0.25);
      expect(searchRequestEnvelope.request.request.fusion).toBe("relative_score");
      yield* replySuccess(
        harness.searchFake,
        searchRequestEnvelope.requestId,
        searchResultsResponse(),
      );

      const result = yield* Fiber.join(searchFiber);
      expect(result.type).toBe("search_results");
    }),
  );

  it.effect("falls back to keyword-only mutable-corpus search when dense state is unavailable", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* SubscriptionRef.set(
        runtime.mutableCorpora,
        new Map([["proof-corpus", persistedMutableCorpusMetadata(2, { hasDenseState: false })]]),
      );
      harness.searchFake.clearOutbound();
      harness.encoderFake.clearOutbound();

      const searchFiber = yield* runtime.searchCorpus(hybridSearchCorpusArgs()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const loadRequest = firstCapturedRequest<LoadMutableCorpusRequestEnvelope>(harness.searchFake);
      yield* replySuccess(
        harness.searchFake,
        loadRequest.requestId,
        loadMutableCorpusResponse(2, { hasDenseState: false }),
      );
      yield* Effect.yieldNow;

      yield* waitForCapturedRequestCount(harness.searchFake, 2);
      expect(harness.encoderFake.capturedRequests()).toHaveLength(0);

      const searchRequestEnvelope =
        harness.searchFake.capturedRequests<SearchRequestEnvelope>()[1];
      expect(searchRequestEnvelope).toBeDefined();
      if (searchRequestEnvelope === undefined) {
        throw new Error("expected mutable keyword fallback request");
      }
      expect(searchRequestEnvelope.request.request.queries).toBeNull();
      expect(searchRequestEnvelope.request.request.text_query).toEqual(["alpha hybrid"]);
      expect(searchRequestEnvelope.request.request.alpha).toBeNull();
      expect(searchRequestEnvelope.request.request.fusion).toBeNull();

      yield* replySuccess(
        harness.searchFake,
        searchRequestEnvelope.requestId,
        searchResultsResponse(),
      );

      const result = yield* Fiber.join(searchFiber);
      expect(result.type).toBe("search_results");
    }),
  );

  it.effect("fails fast for concurrent same-corpus syncs and emits coarse sync lifecycle events", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();
      harness.searchFake.clearOutbound();
      harness.encoderFake.clearOutbound();

      yield* registerMutableCorpus(runtime, harness.searchFake, {
        hasDenseState: true,
      });

      const eventsFiber = yield* runtime.mutableSyncEvents.pipe(
        Stream.take(3),
        Stream.runCollect,
        Effect.forkChild({ startImmediately: true }),
      );

      const firstSyncFiber = yield* runtime.syncCorpus(syncCorpusArgsWithEmbeddings()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const firstRequest = firstCapturedRequest<SyncMutableCorpusRequestEnvelope>(harness.searchFake);
      expect(firstRequest.request.type).toBe("sync_mutable_corpus");
      expect(firstRequest.request.corpus_id).toBe("proof-corpus");

      const secondSyncResult = yield* Effect.result(
        runtime.syncCorpus(syncCorpusArgsWithEmbeddings()),
      );
      expect(secondSyncResult._tag).toBe("Failure");
      if (secondSyncResult._tag !== "Failure") {
        throw new Error("expected concurrent sync to fail fast");
      }
      expect(secondSyncResult.failure.cause).toBe("sync_in_progress");
      expect(harness.searchFake.capturedRequests()).toHaveLength(1);

      yield* replySuccess(
        harness.searchFake,
        firstRequest.requestId,
        syncCorpusResponse({ hasDenseState: true }),
      );

      const firstResult = yield* Fiber.join(firstSyncFiber);
      expect(firstResult.type).toBe("mutable_corpus_synced");
      expect(firstResult.sync.changed).toBe(true);

      const collectedEvents = [...(yield* Fiber.join(eventsFiber))];
      expect(collectedEvents).toHaveLength(3);
      expect(collectedEvents[0]).toEqual({
        type: "sync_started",
        corpusId: "proof-corpus",
        documentCount: 2,
      });
      expect(collectedEvents[1]?.type).toBe("sync_failed");
      if (collectedEvents[1]?.type !== "sync_failed") {
        throw new Error("expected sync_failed event for concurrent sync");
      }
      expect(collectedEvents[1].error.cause).toBe("sync_in_progress");
      expect(collectedEvents[2]).toEqual({
        type: "sync_committed",
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(2, { hasDenseState: true }),
        sync: syncCorpusResponse({ hasDenseState: true }).sync,
      });
    }),
  );

});

layer(makeBrowserRuntimeHarnessLayer())("BrowserSearchRuntime mutable corpus noop events", (it) => {
  it.effect("emits sync_noop when the authoritative snapshot is unchanged", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();
      harness.searchFake.clearOutbound();
      harness.encoderFake.clearOutbound();

      yield* registerMutableCorpus(runtime, harness.searchFake, {
        hasDenseState: true,
      });

      const noopResponse = syncCorpusResponse({
        changed: false,
        documentCount: 2,
        hasDenseState: true,
      });
      const eventsFiber = yield* runtime.mutableSyncEvents.pipe(
        Stream.take(2),
        Stream.runCollect,
        Effect.forkChild({ startImmediately: true }),
      );

      const syncFiber = yield* runtime.syncCorpus(syncCorpusArgsWithEmbeddings()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const request = firstCapturedRequest<SyncMutableCorpusRequestEnvelope>(harness.searchFake);
      expect(request.request.type).toBe("sync_mutable_corpus");
      expect(request.request.corpus_id).toBe("proof-corpus");

      yield* replySuccess(harness.searchFake, request.requestId, noopResponse);

      const result = yield* Fiber.join(syncFiber);
      expect(result.type).toBe("mutable_corpus_synced");
      expect(result.sync.changed).toBe(false);
      expect(result.sync.unchanged).toBe(2);

      const collectedEvents = [...(yield* Fiber.join(eventsFiber))];
      expect(collectedEvents).toEqual([
        {
          type: "sync_started",
          corpusId: "proof-corpus",
          documentCount: 2,
        },
        {
          type: "sync_noop",
          corpusId: "proof-corpus",
          summary: mutableCorpusSummary(2, { hasDenseState: true }),
          sync: noopResponse.sync,
        },
      ]);

      const mutableCorpora = yield* SubscriptionRef.get(runtime.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(2, { hasDenseState: true }),
        loaded: true,
      });
    }),
  );
});

layer(makeBrowserRuntimeHarnessLayer())("BrowserSearchRuntime compatibility preflight", (it) => {
  it.effect("rejects direct search when the query encoder identity mismatches the loaded index", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake);

      const result = yield* Effect.result(
        runtime.searchWithEmbeddings(
          searchRequest(encodedQuery(alternateEncoder()).payload),
        ),
      );
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected incompatible search request to fail");
      }
      expect(result.failure.cause).toBe("encoder_identity_mismatch");
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
    }),
  );

  it.effect("rejects malformed inline embeddings before the search worker sees them", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake);

      const malformedPayload: QueryEmbeddingsPayload = {
        embeddings: [[0.1, 0.2, 0.3]],
        encoder: proofEncoder(),
        dtype: "f32_le",
        layout: "ragged",
      };

      const result = yield* Effect.result(
        runtime.searchWithEmbeddings(searchRequest(malformedPayload)),
      );
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected malformed query payload to fail");
      }
      expect(result.failure.cause).toBe("query_embedding_dim_mismatch");
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
    }),
  );

  it.effect("rejects query payloads that include both inline and binary embeddings", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake);

      const ambiguousPayload: QueryEmbeddingsPayload = {
        embeddings: [[0.1, 0.2, 0.3, 0.4]],
        embeddings_b64: "AQIDBA==",
        shape: [1, 4],
        encoder: proofEncoder(),
        dtype: "f32_le",
        layout: "ragged",
      };

      const result = yield* Effect.result(
        runtime.searchWithEmbeddings(searchRequest(ambiguousPayload)),
      );
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected ambiguous query payload to fail");
      }
      expect(result.failure.cause).toBe("ambiguous_query_embeddings");
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
    }),
  );

  it.effect("rejects binary embeddings without shape metadata before the search worker sees them", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake);

      const malformedPayload: QueryEmbeddingsPayload = {
        embeddings_b64: "AQIDBA==",
        encoder: proofEncoder(),
        dtype: "f32_le",
        layout: "ragged",
      };

      const result = yield* Effect.result(
        runtime.searchWithEmbeddings(searchRequest(malformedPayload)),
      );
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected malformed binary query payload to fail");
      }
      expect(result.failure.cause).toBe("missing_binary_shape");
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
    }),
  );

  it.effect("fails before encode when the initialized encoder does not match the loaded index", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake, proofEncoder());
      yield* initEncoder(encoderClient, harness.encoderFake, alternateEncoder());

      const result = yield* Effect.result(
        runtime.encodeAndSearch({
          text: "alpha",
          searchRequest: {
            type: "search",
            name: "proof-index",
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
          },
        }),
      );
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected encode-and-search compatibility check to fail");
      }
      expect(result.failure.cause).toBe("encoder_identity_mismatch");
      expect(harness.encoderFake.capturedRequests()).toHaveLength(0);
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
    }),
  );

  it.effect("falls back to keyword-only search when the encoder is not initialized but text query exists", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake, proofEncoder());

      const resultFiber = yield* runtime.encodeAndSearch(hybridEncodeAndSearchArgs()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      expect(harness.encoderFake.capturedRequests()).toHaveLength(0);
      yield* waitForCapturedRequestCount(harness.searchFake, 1);
      const searchRequestEnvelope = firstCapturedRequest<SearchRequestEnvelope>(harness.searchFake);
      expect(searchRequestEnvelope.request.type).toBe("search");
      expect(searchRequestEnvelope.request.request.queries).toBeNull();
      expect(searchRequestEnvelope.request.request.text_query).toEqual(["alpha"]);
      expect(searchRequestEnvelope.request.request.alpha).toBeNull();
      expect(searchRequestEnvelope.request.request.fusion).toBeNull();

      yield* replySuccess(
        harness.searchFake,
        searchRequestEnvelope.requestId,
        searchResultsResponse(),
      );
      yield* Effect.yieldNow;

      const result = yield* Fiber.join(resultFiber);
      expect(result.type).toBe("search_results");
      expect(result.results[0]?.document_ids).toEqual([0, 1]);
    }),
  );

});

layer(makeEncodeFailureFallbackLayer())("BrowserSearchRuntime encode failure fallback", (it) => {
  it.effect("falls back to keyword-only search when encode fails after init", () =>
    Effect.gen(function*() {
      const runtime = yield* BrowserSearchRuntime;
      const result = yield* runtime.encodeAndSearch(hybridEncodeAndSearchArgs());
      expect(result.type).toBe("search_results");
      expect(result.results[0]?.document_ids).toEqual([0, 1]);
    }),
  );
});

layer(makeBrowserRuntimeHarnessLayer())("BrowserSearchRuntime malformed encoder output", (it) => {
  it.effect("rejects non-finite encoder output before search handoff", () =>
    Effect.gen(function*() {
      const harness = yield* BrowserRuntimeHarness;
      const searchClient = yield* SearchWorkerClient;
      const encoderClient = yield* EncoderWorkerClient;
      const runtime = yield* BrowserSearchRuntime;

      yield* waitForWorkerStart(harness.searchFake);
      yield* waitForWorkerStart(harness.encoderFake);
      harness.searchFake.dispatchReady();
      harness.encoderFake.dispatchReady();

      yield* loadIndexIntoRuntime(searchClient, harness.searchFake);
      yield* initEncoder(encoderClient, harness.encoderFake);

      const searchFiber = yield* Effect.result(
        runtime.encodeAndSearch({
          text: "alpha",
          searchRequest: {
            type: "search",
            name: "proof-index",
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
          },
        }),
      ).pipe(Effect.forkChild({ startImmediately: true }));
      yield* Effect.yieldNow;

      const encodeRequest =
        firstCapturedRequest<{ readonly type: "encode_query" }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode_query");
      yield* replySuccess(
        harness.encoderFake,
        encodeRequest.requestId,
        malformedEncodeResponseWithNaN(),
      );
      yield* Effect.yieldNow;

      const result = yield* Fiber.join(searchFiber);
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected malformed encoder output to fail encode-and-search");
      }
      expect(result.failure.cause).toBe("decode_failed");
      expect(harness.searchFake.capturedRequests()).toHaveLength(0);
    }),
  );
});
