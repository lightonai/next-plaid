import { expect, it, layer } from "@effect/vitest";
import {
  Context,
  Effect,
  Fiber,
  Layer,
  Scope,
  SubscriptionRef,
} from "effect";
import * as Exit from "effect/Exit";

import type {
  LoadMutableCorpusRequestEnvelope,
  LoadIndexRequestEnvelope,
  RegisterMutableCorpusRequestEnvelope,
  RuntimeErrorResponseEnvelope,
  SearchRequestEnvelope,
  SyncMutableCorpusRequestEnvelope,
} from "../shared/search-contract.js";
import type { SearchClientError } from "./client-errors.js";
import {
  SearchWorkerClient,
  type SearchWorkerState,
} from "./search-worker-client.js";
import {
  type FakeSpawner,
  makeFakeSpawner,
} from "./__tests__/fake-spawner.js";
import * as BrowserWorker from "./browser-worker.js";

interface SearchHarnessApi {
  readonly fake: FakeSpawner;
}

class SearchHarness
  extends Context.Service<SearchHarness, SearchHarnessApi>()(
    "next-plaid-browser/tests/SearchHarness",
  )
{}

function makeSearchHarnessLayer(): Layer.Layer<
  SearchHarness | SearchWorkerClient,
  SearchClientError
> {
  const fake = makeFakeSpawner();
  const workerLayer = BrowserWorker.layer((id) => fake.spawn(id));
  const clientLayer = SearchWorkerClient.layer().pipe(Layer.provide(workerLayer));
  const harnessLayer = Layer.succeed(SearchHarness)(
    SearchHarness.of({
      fake,
    }),
  );

  return Layer.mergeAll(clientLayer, harnessLayer);
}

function searchRequest(): SearchRequestEnvelope {
  return {
    type: "search",
    name: "proof-index",
    request: {
      queries: [],
      params: {
        top_k: 5,
      },
    },
  } as unknown as SearchRequestEnvelope;
}

function runtimeErrorResponse(): RuntimeErrorResponseEnvelope {
  return {
    type: "error",
    code: "internal",
    message: "kernel failed in test",
  };
}

function proofEncoder() {
  return {
    encoder_id: "proof-encoder",
    encoder_build: "proof-build-1",
    embedding_dim: 4,
    normalized: true,
  } as const;
}

function loadIndexRequest(): LoadIndexRequestEnvelope {
  return {
    type: "load_index",
    name: "proof-index",
    encoder: proofEncoder(),
    index: {},
    metadata: null,
    nbits: 2,
    fts_tokenizer: "unicode61",
    max_documents: null,
  } as unknown as LoadIndexRequestEnvelope;
}

function indexLoadedResponse() {
  return {
    type: "index_loaded",
    name: "proof-index",
    summary: {
      name: "proof-index",
      num_documents: 3,
      num_embeddings: 6,
      num_partitions: 2,
      dimension: 4,
      nbits: 2,
      avg_doclen: 2,
      has_metadata: true,
      max_documents: null,
    },
  } as const;
}

function mutableCorpusSummary(documentCount: number) {
  return {
    corpus_id: "proof-corpus",
    document_count: documentCount,
    has_keyword_state: true,
    has_dense_state: false,
    encoder: proofEncoder(),
  } as const;
}

function registerMutableCorpusRequest(): RegisterMutableCorpusRequestEnvelope {
  return {
    type: "register_mutable_corpus",
    corpus_id: "proof-corpus",
    encoder: proofEncoder(),
    fts_tokenizer: "unicode61",
  } as unknown as RegisterMutableCorpusRequestEnvelope;
}

function syncMutableCorpusRequest(): SyncMutableCorpusRequestEnvelope {
  return {
    type: "sync_mutable_corpus",
    corpus_id: "proof-corpus",
    snapshot: {
      documents: [
        {
          document_id: "doc-alpha",
          semantic_text: "alpha semantic body",
          metadata: {
            title: "alpha memo",
            topic: "edge",
          },
        },
      ],
    },
  } as unknown as SyncMutableCorpusRequestEnvelope;
}

function loadMutableCorpusRequest(): LoadMutableCorpusRequestEnvelope {
  return {
    type: "load_mutable_corpus",
    corpus_id: "proof-corpus",
  } as const;
}

function expectFailureState(state: SearchWorkerState): SearchWorkerState & { status: "failed" } {
  expect(state.status).toBe("failed");
  if (state.status !== "failed") {
    throw new Error("expected failed search worker state");
  }
  return state;
}

function waitForWorkerStart(fake: FakeSpawner): Effect.Effect<void> {
  return Effect.gen(function*() {
    while (!fake.isStarted()) {
      yield* Effect.yieldNow;
    }
  });
}

layer(makeSearchHarnessLayer())("SearchWorkerClient worker crash handling", (it) => {
  it.effect("marks the client failed and rejects pending work", () =>
    Effect.gen(function*() {
      const harness = yield* SearchHarness;
      const client = yield* SearchWorkerClient;

      const resultFiber = yield* Effect.result(client.search(searchRequest())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      expect(harness.fake.capturedRequests()).toHaveLength(1);
      harness.fake.dispatchError({ message: "synthetic worker crash" });
      yield* Effect.yieldNow;

      const result = yield* Fiber.join(resultFiber);
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected failed search request");
      }
      expect(result.failure._tag).toBe("TransientClientError");
      expect(result.failure.cause).toBe("worker_crashed");

      const state = expectFailureState(yield* SubscriptionRef.get(client.state));
      expect(state.lastError.cause).toBe("worker_crashed");

      const lateResult = yield* Effect.result(client.search(searchRequest()));
      expect(lateResult._tag).toBe("Failure");
      if (lateResult._tag !== "Failure") {
        throw new Error("expected later search request to fail after worker crash");
      }
      expect(lateResult.failure._tag).toBe("TransientClientError");
      expect(lateResult.failure.cause).toBe("worker_crashed");
      expect(harness.fake.capturedRequests()).toHaveLength(1);
    }),
  );
});

layer(makeSearchHarnessLayer())("SearchWorkerClient typed runtime errors", (it) => {
  it.effect("preserves worker error responses as typed failures", () =>
    Effect.gen(function*() {
      const harness = yield* SearchHarness;
      const client = yield* SearchWorkerClient;

      const resultFiber = yield* Effect.result(client.search(searchRequest())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const request = harness.fake.capturedRequests()[0];
      expect(request).toBeDefined();
      if (request === undefined) {
        throw new Error("expected a captured search request");
      }

      harness.fake.dispatchEnvelope({
        requestId: request.requestId,
        ok: true,
        response: runtimeErrorResponse(),
      });
      yield* Effect.yieldNow;

      const result = yield* Fiber.join(resultFiber);
      expect(result._tag).toBe("Failure");
      if (result._tag !== "Failure") {
        throw new Error("expected failed search request");
      }
      expect(result.failure._tag).toBe("DegradedClientError");
      expect(result.failure.cause).toBe("runtime_error_response");
      expect(result.failure.message).toBe("kernel failed in test");
    }),
  );
});

layer(makeSearchHarnessLayer())("SearchWorkerClient loaded index catalog", (it) => {
  it.effect("records inline-loaded index metadata for wrapper-side compatibility checks", () =>
    Effect.gen(function*() {
      const harness = yield* SearchHarness;
      const client = yield* SearchWorkerClient;

      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();

      const loadFiber = yield* client.loadIndex(loadIndexRequest()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const request = harness.fake.capturedRequests<LoadIndexRequestEnvelope>()[0];
      expect(request).toBeDefined();
      if (request === undefined) {
        throw new Error("expected a captured load-index request");
      }

      harness.fake.dispatchEnvelope({
        requestId: request.requestId,
        ok: true,
        response: indexLoadedResponse(),
      });
      yield* Effect.yieldNow;

      const response = yield* Fiber.join(loadFiber);
      expect(response.type).toBe("index_loaded");

      const loadedIndices = yield* SubscriptionRef.get(client.loadedIndices);
      const metadata = loadedIndices.get("proof-index");
      expect(metadata).toBeDefined();
      expect(metadata?.encoder).toEqual(proofEncoder());
      expect(metadata?.summary.dimension).toBe(4);
      expect(metadata?.source).toBe("load_index");
    }),
  );
});

layer(makeSearchHarnessLayer())("SearchWorkerClient mutable corpus catalog", (it) => {
  it.effect("tracks register, sync, and reload state for mutable corpora", () =>
    Effect.gen(function*() {
      const harness = yield* SearchHarness;
      const client = yield* SearchWorkerClient;

      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();

      const registerFiber = yield* client.registerMutableCorpus(registerMutableCorpusRequest()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const registerRequest =
        harness.fake.capturedRequests<RegisterMutableCorpusRequestEnvelope>()[0];
      expect(registerRequest).toBeDefined();
      if (registerRequest === undefined) {
        throw new Error("expected a captured register-mutable-corpus request");
      }

      harness.fake.dispatchEnvelope({
        requestId: registerRequest.requestId,
        ok: true,
        response: {
          type: "mutable_corpus_registered",
          corpus_id: "proof-corpus",
          created: true,
          summary: mutableCorpusSummary(0),
        },
      });
      yield* Effect.yieldNow;

      const registerResponse = yield* Fiber.join(registerFiber);
      expect(registerResponse.type).toBe("mutable_corpus_registered");

      let mutableCorpora = yield* SubscriptionRef.get(client.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(0),
        loaded: false,
      });

      harness.fake.clearOutbound();

      const syncFiber = yield* client.syncMutableCorpus(syncMutableCorpusRequest()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const syncRequest =
        harness.fake.capturedRequests<SyncMutableCorpusRequestEnvelope>()[0];
      expect(syncRequest).toBeDefined();
      if (syncRequest === undefined) {
        throw new Error("expected a captured sync-mutable-corpus request");
      }
      expect(syncRequest.request.snapshot.documents).toHaveLength(1);

      harness.fake.dispatchEnvelope({
        requestId: syncRequest.requestId,
        ok: true,
        response: {
          type: "mutable_corpus_synced",
          corpus_id: "proof-corpus",
          summary: mutableCorpusSummary(1),
          sync: {
            changed: true,
            added: 1,
            updated: 0,
            deleted: 0,
            unchanged: 0,
          },
        },
      });
      yield* Effect.yieldNow;

      const syncResponse = yield* Fiber.join(syncFiber);
      expect(syncResponse.type).toBe("mutable_corpus_synced");

      mutableCorpora = yield* SubscriptionRef.get(client.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(1),
        loaded: true,
      });

      harness.fake.clearOutbound();

      const loadFiber = yield* client.loadMutableCorpus(loadMutableCorpusRequest()).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const loadRequest =
        harness.fake.capturedRequests<LoadMutableCorpusRequestEnvelope>()[0];
      expect(loadRequest).toBeDefined();
      if (loadRequest === undefined) {
        throw new Error("expected a captured load-mutable-corpus request");
      }
      expect(loadRequest.request.corpus_id).toBe("proof-corpus");

      harness.fake.dispatchEnvelope({
        requestId: loadRequest.requestId,
        ok: true,
        response: {
          type: "mutable_corpus_loaded",
          corpus_id: "proof-corpus",
          summary: mutableCorpusSummary(1),
        },
      });
      yield* Effect.yieldNow;

      const loadResponse = yield* Fiber.join(loadFiber);
      expect(loadResponse.type).toBe("mutable_corpus_loaded");

      mutableCorpora = yield* SubscriptionRef.get(client.mutableCorpora);
      expect(mutableCorpora.get("proof-corpus")).toEqual({
        corpusId: "proof-corpus",
        summary: mutableCorpusSummary(1),
        loaded: true,
      });
    }),
  );
});

it.effect("fails immediately after scope close", () =>
  Effect.gen(function*() {
    const scope = yield* Scope.make();
    const context = yield* Layer.buildWithScope(makeSearchHarnessLayer(), scope);
    const client = Context.get(context, SearchWorkerClient);

    yield* Scope.close(scope, Exit.void);

    const state = yield* SubscriptionRef.get(client.state);
    expect(state.status).toBe("disposed");

    const result = yield* Effect.result(client.search(searchRequest()));
    expect(result._tag).toBe("Failure");
    if (result._tag !== "Failure") {
      throw new Error("expected disposed search request to fail");
    }
    expect(result.failure._tag).toBe("TransientClientError");
    expect(result.failure.cause).toBe("worker_disposed");
  }),
);
