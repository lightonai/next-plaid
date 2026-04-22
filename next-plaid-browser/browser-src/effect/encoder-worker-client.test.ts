import { describe, expect, it, layer } from "@effect/vitest";
import {
  Context,
  Effect,
  Fiber,
  Layer,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect";
import * as Exit from "effect/Exit";

import type {
  EncodedDocument,
  EncodedQuery,
  EncoderCapabilities,
  EncoderCreateInput,
  EncoderInitEvent,
} from "../model-worker/types.js";
import type { EncoderClientError } from "./client-errors.js";
import {
  type EncoderWorkerClientApi,
  EncoderWorkerClient,
} from "./encoder-worker-client.js";
import {
  type CapturedRequest,
  type FakeSpawner,
  makeFakeSpawner,
} from "./__tests__/fake-spawner.js";
import * as BrowserWorker from "./browser-worker.js";

interface EncoderHarnessApi {
  readonly fake: FakeSpawner;
}

class EncoderHarness
  extends Context.Service<EncoderHarness, EncoderHarnessApi>()(
    "next-plaid-browser/tests/EncoderHarness",
  )
{}

function makeEncoderHarnessLayer(): Layer.Layer<
  EncoderHarness | EncoderWorkerClient,
  EncoderClientError
> {
  const fake = makeFakeSpawner();
  const workerLayer = BrowserWorker.layer((id) => fake.spawn(id));
  const clientLayer = EncoderWorkerClient.layer().pipe(Layer.provide(workerLayer));
  const harnessLayer = Layer.succeed(EncoderHarness)(
    EncoderHarness.of({
      fake,
    }),
  );

  return Layer.mergeAll(clientLayer, harnessLayer);
}

function encoderInput(): EncoderCreateInput {
  return {
    encoder: {
      encoder_id: "proof-encoder",
      encoder_build: "build-1",
      embedding_dim: 4,
      normalized: true,
    },
    modelUrl: "/proof/model.onnx",
    onnxConfigUrl: "/proof/onnx_config.json",
    tokenizerUrl: "/proof/tokenizer.json",
    prefer: "wasm",
  };
}

function encoderCapabilities(): EncoderCapabilities {
  return {
    backend: "wasm",
    threaded: false,
    persistentStorage: true,
    encoderId: "proof-encoder",
    encoderBuild: "build-1",
    embeddingDim: 4,
    queryLength: 8,
    doQueryExpansion: false,
    normalized: true,
  };
}

function initResponse() {
  return {
    type: "encoder_ready",
    state: "ready",
    capabilities: encoderCapabilities(),
  } as const;
}

function encodedQuery(): EncodedQuery {
  return {
    payload: {
      embeddings: [[0.1, 0.2, 0.3, 0.4]],
      encoder: encoderInput().encoder,
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

function encodeResponse() {
  return {
    type: "encoded_query",
    encoded: encodedQuery(),
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

function encodeDocumentResponse() {
  return {
    type: "encoded_document",
    encoded: encodedDocument(),
  } as const;
}

function initEvents(): ReadonlyArray<EncoderInitEvent> {
  return [
    {
      stage: "asset_cache_miss",
      url: encoderInput().modelUrl,
    },
    {
      stage: "asset_fetch_start",
      url: encoderInput().modelUrl,
      expectedBytes: 1024,
    },
    {
      stage: "asset_fetch_complete",
      url: encoderInput().modelUrl,
      bytesReceived: 1024,
    },
    {
      stage: "asset_cache_miss",
      url: encoderInput().tokenizerUrl,
    },
    {
      stage: "asset_fetch_start",
      url: encoderInput().tokenizerUrl,
      expectedBytes: 1024,
    },
    {
      stage: "asset_fetch_complete",
      url: encoderInput().tokenizerUrl,
      bytesReceived: 1024,
    },
    {
      stage: "asset_cache_miss",
      url: encoderInput().onnxConfigUrl,
    },
    {
      stage: "asset_fetch_start",
      url: encoderInput().onnxConfigUrl,
      expectedBytes: 1024,
    },
    {
      stage: "asset_fetch_complete",
      url: encoderInput().onnxConfigUrl,
      bytesReceived: 1024,
    },
    {
      stage: "config_validated",
      queryLength: 8,
      embeddingDim: 4,
    },
    {
      stage: "session_create_start",
    },
    {
      stage: "session_create_complete",
      durationMs: 11,
    },
    {
      stage: "warmup_start",
    },
    {
      stage: "warmup_complete",
      durationMs: 7,
    },
    {
      stage: "ready",
      capabilities: encoderCapabilities(),
    },
  ];
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

function waitForCapturedRequestType<TRequest extends { readonly type: string }>(
  fake: FakeSpawner,
  requestType: TRequest["type"],
): Effect.Effect<CapturedRequest<TRequest>> {
  return Effect.gen(function*() {
    while (true) {
      const request = fake
        .capturedRequests<TRequest>()
        .find((captured) => captured.request.type === requestType);
      if (request !== undefined) {
        return request;
      }
      yield* Effect.yieldNow;
    }
  });
}

function initEncoderClient(
  harness: EncoderHarnessApi,
  client: EncoderWorkerClientApi,
): Effect.Effect<void, EncoderClientError> {
  return Effect.gen(function*() {
    const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
      Effect.forkChild({ startImmediately: true }),
    );
    yield* Effect.yieldNow;
    yield* waitForWorkerStart(harness.fake);
    harness.fake.dispatchReady();
    yield* Effect.yieldNow;

    const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
    harness.fake.dispatchEnvelope({
      requestId: initRequest.requestId,
      ok: true,
      response: initResponse(),
    });

    const initResult = yield* Fiber.join(initFiber);
    expect(initResult._tag).toBe("Success");
    if (initResult._tag !== "Success") {
      throw new Error("expected successful init result");
    }
    expect(initResult.success).toEqual(encoderCapabilities());
  });
}

function encodeThroughWorker(
  harness: EncoderHarnessApi,
  client: EncoderWorkerClientApi,
  text: string,
): Effect.Effect<void, EncoderClientError> {
  return Effect.gen(function*() {
    const beforeCount = harness.fake.capturedRequests().length;
    const encodeFiber = yield* Effect.result(client.encodeQuery({ text })).pipe(
      Effect.forkChild({ startImmediately: true }),
    );
    yield* Effect.yieldNow;

    const encodeRequest = yield* waitForCapturedRequestType<
      { readonly type: "encode_query" }
    >(harness.fake, "encode_query");
    expect(harness.fake.capturedRequests()).toHaveLength(beforeCount + 1);

    harness.fake.dispatchEnvelope({
      requestId: encodeRequest.requestId,
      ok: true,
      response: encodeResponse(),
    });
    yield* Effect.yieldNow;

    const encodeResult = yield* Fiber.join(encodeFiber);
    expect(encodeResult._tag).toBe("Success");
    if (encodeResult._tag !== "Success") {
      throw new Error("expected successful encode result");
    }
    expect(encodeResult.success).toEqual(encodedQuery());
  });
}

layer(makeEncoderHarnessLayer())("EncoderWorkerClient initial init lifecycle", (it) => {
  it.effect("spawns one worker lifetime and reuses the completed init result", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;

      yield* waitForWorkerStart(harness.fake);
      expect(harness.fake.spawnCount()).toBe(1);

      const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);

      expect(harness.fake.capturedRequests()).toHaveLength(0);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
      expect(initRequest.request.type).toBe("init");

      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: true,
        response: initResponse(),
      });

      const firstResult = yield* Fiber.join(initFiber);
      expect(firstResult._tag).toBe("Success");
      if (firstResult._tag !== "Success") {
        throw new Error("expected successful encoder init");
      }
      expect(firstResult.success).toEqual(encoderCapabilities());

      const repeatResult = yield* Effect.result(client.init(encoderInput()));
      expect(repeatResult._tag).toBe("Success");
      if (repeatResult._tag !== "Success") {
        throw new Error("expected repeated init to reuse prior result");
      }
      expect(repeatResult.success).toEqual(encoderCapabilities());
      expect(harness.fake.capturedRequests()).toHaveLength(1);
    }),
  );
});

layer(makeEncoderHarnessLayer())("EncoderWorkerClient concurrent init callers", (it) => {
  it.effect("shares one in-flight init result", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;

      const firstFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      const secondFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);

      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      expect(harness.fake.capturedRequests()).toHaveLength(1);
      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: true,
        response: initResponse(),
      });

      const firstResult = yield* Fiber.join(firstFiber);
      const secondResult = yield* Fiber.join(secondFiber);
      expect(firstResult).toEqual(secondResult);
      expect(firstResult._tag).toBe("Success");
      if (firstResult._tag !== "Success") {
        throw new Error("expected successful shared init result");
      }
      expect(firstResult.success).toEqual(encoderCapabilities());
    }),
  );
});

layer(makeEncoderHarnessLayer())("EncoderWorkerClient init failure fan-out", (it) => {
  it.effect("fails all joined init callers when the init request fails", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;

      const firstFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      const secondFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);

      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      expect(harness.fake.capturedRequests()).toHaveLength(1);
      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);

      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: false,
        error: "synthetic init failure",
      });
      yield* Effect.yieldNow;

      const firstResult = yield* Fiber.join(firstFiber);
      const secondResult = yield* Fiber.join(secondFiber);
      expect(firstResult._tag).toBe("Failure");
      expect(secondResult._tag).toBe("Failure");
      if (firstResult._tag !== "Failure" || secondResult._tag !== "Failure") {
        throw new Error("expected both joined init callers to fail");
      }

      expect(firstResult.failure._tag).toBe("DegradedClientError");
      expect(secondResult.failure._tag).toBe("DegradedClientError");
      expect(firstResult.failure.cause).toBe("worker_failure_envelope");
      expect(secondResult.failure.cause).toBe("worker_failure_envelope");

      const state = yield* SubscriptionRef.get(client.state);
      expect(state.status).toBe("failed");
    }),
  );
});

layer(makeEncoderHarnessLayer())("EncoderWorkerClient readiness gating", (it) => {
  it.effect("holds encode until init opens the ready gate", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;

      const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
      const encodeFiber = yield* Effect.result(client.encodeQuery({ text: "alpha" })).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: true,
        response: initResponse(),
      });
      yield* Effect.yieldNow;

      expect(harness.fake.capturedRequests()).toHaveLength(2);
      const encodeRequest =
        harness.fake.capturedRequests<{ readonly type: "encode_query" }>()[1];
      expect(encodeRequest).toBeDefined();
      if (encodeRequest === undefined) {
        throw new Error("expected a captured encode request");
      }
      expect(encodeRequest.request.type).toBe("encode_query");

      harness.fake.dispatchEnvelope({
        requestId: encodeRequest.requestId,
        ok: true,
        response: encodeResponse(),
      });

      const initResult = yield* Fiber.join(initFiber);
      const encodeResult = yield* Fiber.join(encodeFiber);
      expect(initResult._tag).toBe("Success");
      if (initResult._tag !== "Success") {
        throw new Error("expected successful init result");
      }
      expect(initResult.success).toEqual(encoderCapabilities());

      expect(encodeResult._tag).toBe("Success");
      if (encodeResult._tag !== "Success") {
        throw new Error("expected successful encode result");
      }
      expect(encodeResult.success).toEqual(encodedQuery());
    }),
  );

  it.effect("routes document encoding through a distinct worker request", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;

      yield* initEncoderClient(harness, client);

      const encodeFiber = yield* Effect.result(client.encodeDocument({ text: "alpha beta" })).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      const encodeRequest = yield* waitForCapturedRequestType<{
        readonly type: "encode_document";
        readonly payload: { readonly text: string };
      }>(harness.fake, "encode_document");
      expect(encodeRequest).toBeDefined();
      if (encodeRequest === undefined) {
        throw new Error("expected one captured document encode request");
      }
      expect(encodeRequest.request.type).toBe("encode_document");
      expect(encodeRequest.request.payload.text).toBe("alpha beta");

      harness.fake.dispatchEnvelope({
        requestId: encodeRequest.requestId,
        ok: true,
        response: encodeDocumentResponse(),
      });

      const encodeResult = yield* Fiber.join(encodeFiber);
      expect(encodeResult._tag).toBe("Success");
      if (encodeResult._tag !== "Success") {
        throw new Error("expected successful document encode result");
      }
      expect(encodeResult.success).toEqual(encodedDocument());
    }),
  );
});

layer(
  makeEncoderHarnessLayer(),
)("EncoderWorkerClient queued encode failure state", (it) => {
  it.effect("preserves ready capabilities when an encode waiting on init fails", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;

      const failedEventFiber = yield* client.events.pipe(
        Stream.filter((event) => event.stage === "failed"),
        Stream.take(1),
        Stream.runCollect,
        Effect.forkChild({ startImmediately: true }),
      );
      const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
      const encodeFiber = yield* Effect.result(client.encodeQuery({ text: "beta" })).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;

      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: true,
        response: initResponse(),
      });
      yield* Effect.yieldNow;

      const encodeRequest = yield* waitForCapturedRequestType<
        { readonly type: "encode_query" }
      >(harness.fake, "encode_query");

      harness.fake.dispatchEnvelope({
        requestId: encodeRequest.requestId,
        ok: false,
        error: "synthetic encode failure",
      });
      yield* Effect.yieldNow;

      const initResult = yield* Fiber.join(initFiber);
      expect(initResult._tag).toBe("Success");
      if (initResult._tag !== "Success") {
        throw new Error("expected successful init result");
      }
      expect(initResult.success).toEqual(encoderCapabilities());

      const encodeResult = yield* Fiber.join(encodeFiber);
      expect(encodeResult._tag).toBe("Failure");
      if (encodeResult._tag !== "Failure") {
        throw new Error("expected queued encode to fail");
      }
      expect(encodeResult.failure._tag).toBe("DegradedClientError");
      expect(encodeResult.failure.cause).toBe("worker_failure_envelope");

      const failedEvents = [...(yield* Fiber.join(failedEventFiber))];
      expect(failedEvents).toHaveLength(1);
      const failedEvent = failedEvents[0];
      expect(failedEvent).toBeDefined();
      if (failedEvent === undefined) {
        throw new Error("expected one failed lifecycle event");
      }
      expect(failedEvent.stage).toBe("failed");
      if (failedEvent.stage !== "failed") {
        throw new Error("expected a failed lifecycle event");
      }
      expect(failedEvent.error.cause).toBe("worker_failure_envelope");

      const state = yield* SubscriptionRef.get(client.state);
      expect(state.status).toBe("failed");
      if (state.status !== "failed") {
        throw new Error("expected failed encoder state");
      }
      expect(state.capabilities).toEqual(encoderCapabilities());
      expect(state.lastError.cause).toBe("worker_failure_envelope");
    }),
  );
});

layer(makeEncoderHarnessLayer())("EncoderWorkerClient init events", (it) => {
  it.effect("emits init lifecycle events in a stable order", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;
      const expectedEvents = initEvents();

      const eventsFiber = yield* client.events.pipe(
        Stream.take(expectedEvents.length),
        Stream.runCollect,
        Effect.forkChild({ startImmediately: true }),
      );
      const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);

      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
      for (const event of expectedEvents) {
        harness.fake.dispatchEnvelope({
          requestId: initRequest.requestId,
          ok: true,
          event,
        });
      }
      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: true,
        response: initResponse(),
      });

      const collectedEvents = [...(yield* Fiber.join(eventsFiber))];
      const initResult = yield* Fiber.join(initFiber);
      expect(collectedEvents).toEqual(expectedEvents);
      expect(initResult._tag).toBe("Success");
      if (initResult._tag !== "Success") {
        throw new Error("expected successful init result");
      }
      expect(initResult.success).toEqual(encoderCapabilities());
    }),
  );

  it.effect("broadcasts init lifecycle events to multiple subscribers", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;
      const expectedEvents = initEvents();

      const firstEventsFiber = yield* client.events.pipe(
        Stream.take(expectedEvents.length),
        Stream.runCollect,
        Effect.forkChild({ startImmediately: true }),
      );
      const secondEventsFiber = yield* client.events.pipe(
        Stream.take(expectedEvents.length),
        Stream.runCollect,
        Effect.forkChild({ startImmediately: true }),
      );
      const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);

      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
      for (const event of expectedEvents) {
        harness.fake.dispatchEnvelope({
          requestId: initRequest.requestId,
          ok: true,
          event,
        });
      }
      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: true,
        response: initResponse(),
      });

      const firstEvents = [...(yield* Fiber.join(firstEventsFiber))];
      const secondEvents = [...(yield* Fiber.join(secondEventsFiber))];
      const initResult = yield* Fiber.join(initFiber);
      expect(firstEvents).toEqual(expectedEvents);
      expect(secondEvents).toEqual(expectedEvents);
      expect(initResult._tag).toBe("Success");
      if (initResult._tag !== "Success") {
        throw new Error("expected successful init result");
      }
    }),
  );
});

layer(makeEncoderHarnessLayer())("EncoderWorkerClient terminal failure events", (it) => {
  it.effect("publishes a failed event when the worker crashes after ready", () =>
    Effect.gen(function*() {
      const harness = yield* EncoderHarness;
      const client = yield* EncoderWorkerClient;

      const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);

      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
      harness.fake.dispatchEnvelope({
        requestId: initRequest.requestId,
        ok: true,
        response: initResponse(),
      });

      const initResult = yield* Fiber.join(initFiber);
      expect(initResult._tag).toBe("Success");
      if (initResult._tag !== "Success") {
        throw new Error("expected successful init result");
      }

      const failedEventFiber = yield* client.events.pipe(
        Stream.filter((event) => event.stage === "failed"),
        Stream.take(1),
        Stream.runCollect,
        Effect.forkChild({ startImmediately: true }),
      );

      harness.fake.dispatchError({ message: "synthetic encoder crash" });
      yield* Effect.yieldNow;

      const failedEvents = [...(yield* Fiber.join(failedEventFiber))];
      expect(failedEvents).toHaveLength(1);
      const failedEvent = failedEvents[0];
      expect(failedEvent).toBeDefined();
      if (failedEvent === undefined) {
        throw new Error("expected one failed lifecycle event");
      }
      expect(failedEvent.stage).toBe("failed");
      if (failedEvent.stage !== "failed") {
        throw new Error("expected a failed lifecycle event");
      }
      expect(failedEvent.error.cause).toBe("worker_crashed");

      const state = yield* SubscriptionRef.get(client.state);
      expect(state.status).toBe("failed");
    }),
  );
});

it.effect("marks the client disposed and rejects new calls after scope close", () =>
  Effect.gen(function*() {
    const scope = yield* Scope.make();
    const context = yield* Layer.buildWithScope(makeEncoderHarnessLayer(), scope);
    const client = Context.get(context, EncoderWorkerClient);

    yield* Scope.close(scope, Exit.void);

    const state = yield* SubscriptionRef.get(client.state);
    expect(state.status).toBe("disposed");

    const encodeResult = yield* Effect.result(client.encodeQuery({ text: "alpha" }));
    expect(encodeResult._tag).toBe("Failure");
    if (encodeResult._tag !== "Failure") {
      throw new Error("expected disposed encoder encode call to fail");
    }
    expect(encodeResult.failure._tag).toBe("PermanentClientError");
    expect(encodeResult.failure.cause).toBe("encoder_disposed");

    const initResult = yield* Effect.result(client.init(encoderInput()));
    expect(initResult._tag).toBe("Failure");
    if (initResult._tag !== "Failure") {
      throw new Error("expected disposed encoder init call to fail");
    }
    expect(initResult.failure._tag).toBe("PermanentClientError");
    expect(initResult.failure.cause).toBe("encoder_disposed");
  }),
);

it.effect("fails joined init callers when the client scope closes mid-init", () =>
  Effect.gen(function*() {
    const scope = yield* Scope.make();
    const context = yield* Layer.buildWithScope(makeEncoderHarnessLayer(), scope);
    const harness = Context.get(context, EncoderHarness);
    const client = Context.get(context, EncoderWorkerClient);

    const firstFiber = yield* Effect.result(client.init(encoderInput())).pipe(
      Effect.forkChild({ startImmediately: true }),
    );
    const secondFiber = yield* Effect.result(client.init(encoderInput())).pipe(
      Effect.forkChild({ startImmediately: true }),
    );
    yield* Effect.yieldNow;
    yield* waitForWorkerStart(harness.fake);
    harness.fake.dispatchReady();
    yield* Effect.yieldNow;

    expect(harness.fake.capturedRequests()).toHaveLength(1);
    yield* Scope.close(scope, Exit.void);

    const firstResult = yield* Fiber.join(firstFiber);
    const secondResult = yield* Fiber.join(secondFiber);
    expect(firstResult._tag).toBe("Failure");
    expect(secondResult._tag).toBe("Failure");
    if (firstResult._tag !== "Failure" || secondResult._tag !== "Failure") {
      throw new Error("expected joined init callers to fail when scope closes");
    }
    expect(firstResult.failure.cause).toBe("encoder_disposed");
    expect(secondResult.failure.cause).toBe("encoder_disposed");

    const state = yield* SubscriptionRef.get(client.state);
    expect(state.status).toBe("disposed");
  }),
);

describe("Deferred wrapper invariants", () => {
  layer(makeEncoderHarnessLayer())(
    "EncoderWorkerClient encode cache",
    (it) => {
      it.effect("concurrent duplicate encode(text) calls share one in-flight result", () =>
        Effect.gen(function*() {
          const harness = yield* EncoderHarness;
          const client = yield* EncoderWorkerClient;

          const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
            Effect.forkChild({ startImmediately: true }),
          );
          yield* Effect.yieldNow;
          yield* waitForWorkerStart(harness.fake);
          harness.fake.dispatchReady();
          yield* Effect.yieldNow;

          const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
          harness.fake.dispatchEnvelope({
            requestId: initRequest.requestId,
            ok: true,
            response: initResponse(),
          });
          const initResult = yield* Fiber.join(initFiber);
          expect(initResult._tag).toBe("Success");
          if (initResult._tag !== "Success") {
            throw new Error("expected successful init result");
          }

          const firstFiber = yield* Effect.result(client.encodeQuery({ text: "alpha" })).pipe(
            Effect.forkChild({ startImmediately: true }),
          );
          const secondFiber = yield* Effect.result(client.encodeQuery({ text: "alpha" })).pipe(
            Effect.forkChild({ startImmediately: true }),
          );
          yield* Effect.yieldNow;

          expect(harness.fake.capturedRequests()).toHaveLength(2);
          const encodeRequest =
            harness.fake.capturedRequests<{ readonly type: "encode_query" }>()[1];
          expect(encodeRequest).toBeDefined();
          if (encodeRequest === undefined) {
            throw new Error("expected one captured encode request");
          }
          expect(encodeRequest.request.type).toBe("encode_query");

          harness.fake.dispatchEnvelope({
            requestId: encodeRequest.requestId,
            ok: true,
            response: encodeResponse(),
          });

          const firstResult = yield* Fiber.join(firstFiber);
          const secondResult = yield* Fiber.join(secondFiber);
          expect(firstResult).toEqual(secondResult);
          expect(firstResult._tag).toBe("Success");
          if (firstResult._tag !== "Success") {
            throw new Error("expected successful shared encode result");
          }
          expect(firstResult.success).toEqual(encodedQuery());
          expect(harness.fake.capturedRequests()).toHaveLength(2);
        }),
      );

      it.effect("repeated successful encode(text) hits the completed-query cache", () =>
        Effect.gen(function*() {
          const harness = yield* EncoderHarness;
          const client = yield* EncoderWorkerClient;

          const initFiber = yield* Effect.result(client.init(encoderInput())).pipe(
            Effect.forkChild({ startImmediately: true }),
          );
          yield* Effect.yieldNow;
          yield* waitForWorkerStart(harness.fake);
          harness.fake.dispatchReady();
          yield* Effect.yieldNow;

          const initRequest = firstCapturedRequest<{ readonly type: "init" }>(harness.fake);
          harness.fake.dispatchEnvelope({
            requestId: initRequest.requestId,
            ok: true,
            response: initResponse(),
          });
          const initResult = yield* Fiber.join(initFiber);
          expect(initResult._tag).toBe("Success");
          if (initResult._tag !== "Success") {
            throw new Error("expected successful init result");
          }

          const firstEncodeFiber = yield* Effect.result(client.encodeQuery({ text: "alpha" })).pipe(
            Effect.forkChild({ startImmediately: true }),
          );
          yield* Effect.yieldNow;

          expect(harness.fake.capturedRequests()).toHaveLength(2);
          const encodeRequest =
            harness.fake.capturedRequests<{ readonly type: "encode_query" }>()[1];
          expect(encodeRequest).toBeDefined();
          if (encodeRequest === undefined) {
            throw new Error("expected one captured encode request");
          }

          harness.fake.dispatchEnvelope({
            requestId: encodeRequest.requestId,
            ok: true,
            response: encodeResponse(),
          });
          yield* Effect.yieldNow;

          const firstResult = yield* Fiber.join(firstEncodeFiber);
          expect(firstResult._tag).toBe("Success");
          if (firstResult._tag !== "Success") {
            throw new Error("expected successful first encode");
          }
          expect(firstResult.success).toEqual(encodedQuery());

          const secondResult = yield* Effect.result(client.encodeQuery({ text: "alpha" }));
          expect(secondResult._tag).toBe("Success");
          if (secondResult._tag !== "Success") {
            throw new Error("expected successful repeated cached encode result");
          }
          expect(secondResult.success).toEqual(encodedQuery());
          expect(harness.fake.capturedRequests()).toHaveLength(2);
        }),
      );
    },
  );

  it.effect("model reload clears the query cache", () =>
    Effect.gen(function*() {
      const firstScope = yield* Scope.make();
      try {
        const firstContext = yield* Layer.buildWithScope(
          makeEncoderHarnessLayer(),
          firstScope,
        );
        const firstHarness = Context.get(firstContext, EncoderHarness);
        const firstClient = Context.get(firstContext, EncoderWorkerClient);

        yield* initEncoderClient(firstHarness, firstClient);
        yield* encodeThroughWorker(firstHarness, firstClient, "alpha");
        expect(firstHarness.fake.capturedRequests()).toHaveLength(2);
      } finally {
        yield* Scope.close(firstScope, Exit.void);
      }

      const secondScope = yield* Scope.make();
      try {
        const secondContext = yield* Layer.buildWithScope(
          makeEncoderHarnessLayer(),
          secondScope,
        );
        const secondHarness = Context.get(secondContext, EncoderHarness);
        const secondClient = Context.get(secondContext, EncoderWorkerClient);

        yield* initEncoderClient(secondHarness, secondClient);
        yield* encodeThroughWorker(secondHarness, secondClient, "alpha");
        expect(secondHarness.fake.capturedRequests()).toHaveLength(2);
      } finally {
        yield* Scope.close(secondScope, Exit.void);
      }
    }),
  );

});
