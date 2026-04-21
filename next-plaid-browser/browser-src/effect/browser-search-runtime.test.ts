import { expect, layer } from "@effect/vitest";
import { Context, Effect, Fiber, Layer, Stream, SubscriptionRef } from "effect";

import type {
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
import type {
  EncoderStateSnapshot,
  EncoderWorkerClientApi,
} from "./encoder-worker-client.js";
import type {
  EncoderIdentity,
  LoadIndexRequestEnvelope,
  QueryEmbeddingsPayload,
  SearchRequestEnvelope,
  SearchResultsResponseEnvelope,
} from "../shared/search-contract.js";
import { BrowserSearchRuntime } from "./browser-search-runtime.js";
import * as BrowserWorker from "./browser-worker.js";
import { EncoderWorkerClient } from "./encoder-worker-client.js";
import type {
  LoadedSearchIndexMetadata,
  MutableCorpusMetadata,
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
  const searchLayer = Layer.effect(SearchWorkerClient)(
    Effect.gen(function*() {
      const state = yield* SubscriptionRef.make<SearchWorkerState>({
        status: "ready" as const,
        lastError: null,
      });
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

      return SearchWorkerClient.of({
        state,
        loadedIndices,
        mutableCorpora: yield* SubscriptionRef.make<
          ReadonlyMap<string, MutableCorpusMetadata>
        >(new Map()),
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
        encode: () =>
          Effect.fail(
            permanentClientError({
              cause: "synthetic_encode_failure",
              message: "encode failed in fallback test",
              operation: "encoder_worker.encode",
              details: null,
            }),
          ),
      });
    }),
  );

  const appLayer = Layer.mergeAll(searchLayer, encoderLayer);
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
    doQueryExpansion: false,
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
      },
    ],
    num_queries: 1,
    timing: null,
  };
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

    const request = firstCapturedRequest<{ readonly type: "init" }>(fake);
    expect(request.request.type).toBe("init");
    yield* replySuccess(fake, request.requestId, initResponse(encoder));
    const capabilities = yield* Fiber.join(initFiber);
    expect(capabilities).toEqual(encoderCapabilities(encoder));

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

      const encodeRequest = firstCapturedRequest<{ readonly type: "encode" }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode");
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

      const encodeRequest = firstCapturedRequest<{ readonly type: "encode" }>(harness.encoderFake);
      expect(encodeRequest.request.type).toBe("encode");
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
