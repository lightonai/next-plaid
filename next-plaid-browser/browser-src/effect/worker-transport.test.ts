import { expect, it, layer } from "@effect/vitest";
import {
  Context,
  Duration,
  Effect,
  Fiber,
  Layer,
  Schema,
  Scope,
} from "effect";
import * as Exit from "effect/Exit";
import { TestClock } from "effect/testing";

import {
  permanentClientError,
  type SearchClientError,
} from "./client-errors.js";
import {
  type WorkerTransport,
  makeWorkerTransport,
} from "./worker-transport.js";
import {
  type CapturedRequest,
  type FakeSpawner,
  makeFakeSpawner,
} from "./__tests__/fake-spawner.js";
import * as BrowserWorker from "./browser-worker.js";

interface TransportHarnessApi {
  readonly fake: FakeSpawner;
  readonly transport: WorkerTransport<{ readonly type: "ping" }>;
}

class TransportHarness
  extends Context.Service<TransportHarness, TransportHarnessApi>()(
    "next-plaid-browser/tests/TransportHarness",
  )
{}

function decodeStringResponse(
  value: unknown,
): Effect.Effect<string, SearchClientError> {
  return Schema.decodeUnknownEffect(Schema.String)(value).pipe(
    Effect.mapError((error) =>
      permanentClientError({
        cause: "decode_failed",
        message: `failed to decode string response: ${String(error)}`,
        operation: "transport_test.ping",
        details: error,
      }),
    ),
  );
}

function makeTransportHarnessLayer(
  requestTimeout: Duration.Input = Duration.seconds(5),
): Layer.Layer<TransportHarness, SearchClientError> {
  const fake = makeFakeSpawner();
  const workerLayer = BrowserWorker.layer((id) => fake.spawn(id));

  return Layer.effect(
    TransportHarness,
    makeWorkerTransport<{ readonly type: "ping" }>({
      workerKind: "search",
      requestTimeout,
    }).pipe(
      Effect.map((transport) =>
        TransportHarness.of({
          fake,
          transport,
        }),
      ),
    ),
  ).pipe(Layer.provide(workerLayer));
}

function singleCapturedRequest<TRequest = unknown>(
  fake: FakeSpawner,
): CapturedRequest<TRequest> {
  const requests = fake.capturedRequests<TRequest>();
  expect(requests).toHaveLength(1);
  const request = requests[0];
  if (request === undefined) {
    throw new Error("expected one captured worker request");
  }
  return request;
}

function expectFailure(
  result:
    | { readonly _tag: "Success"; readonly success: string }
    | { readonly _tag: "Failure"; readonly failure: SearchClientError },
  tag: SearchClientError["_tag"],
  cause: string,
): SearchClientError {
  expect(result._tag).toBe("Failure");
  if (result._tag !== "Failure") {
    throw new Error("expected failure result");
  }
  expect(result.failure._tag).toBe(tag);
  expect(result.failure.cause).toBe(cause);
  return result.failure;
}

function ping(
  transport: WorkerTransport<{ readonly type: "ping" }>,
): Effect.Effect<string, SearchClientError> {
  return transport.request(
    { type: "ping" },
    {
      operation: "transport_test.ping",
      requestType: "ping",
      decodeResponse: decodeStringResponse,
    },
  );
}

function waitForWorkerStart(fake: FakeSpawner): Effect.Effect<void> {
  return Effect.gen(function*() {
    while (!fake.isStarted()) {
      yield* Effect.yieldNow;
    }
  });
}

layer(makeTransportHarnessLayer())("WorkerTransport malformed envelope handling", (it) => {
  it.effect("fails the caller with a permanent decode error", () =>
    Effect.gen(function*() {
      const harness = yield* TransportHarness;

      const resultFiber = yield* Effect.result(ping(harness.transport)).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const request = singleCapturedRequest<{ readonly type: "ping" }>(harness.fake);
      harness.fake.dispatchEnvelope({
        requestId: request.requestId,
        ok: "broken",
      });
      yield* Effect.yieldNow;

      const result = yield* Fiber.join(resultFiber);
      const failure = expectFailure(result, "PermanentClientError", "decode_failed");
      expect(failure.requestId).toBe(request.requestId);
    }),
  );
});

layer(
  makeTransportHarnessLayer(Duration.seconds(3)),
)("WorkerTransport timeout handling", (it) => {
  it.effect("maps timeouts to typed transient failures", () =>
    Effect.gen(function*() {
      const harness = yield* TransportHarness;

      const resultFiber = yield* Effect.result(ping(harness.transport)).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      const request = singleCapturedRequest<{ readonly type: "ping" }>(harness.fake);
      yield* TestClock.adjust(Duration.seconds(3));

      const result = yield* Fiber.join(resultFiber);
      const failure = expectFailure(result, "TransientClientError", "timeout");
      expect(failure.requestId).toBe(request.requestId);
    }),
  );
});

layer(makeTransportHarnessLayer())("WorkerTransport messageerror handling", (it) => {
  it.effect("maps messageerror to a typed transient failure and rejects later calls", () =>
    Effect.gen(function*() {
      const harness = yield* TransportHarness;

      const resultFiber = yield* Effect.result(ping(harness.transport)).pipe(
        Effect.forkChild({ startImmediately: true }),
      );
      yield* Effect.yieldNow;
      yield* waitForWorkerStart(harness.fake);
      harness.fake.dispatchReady();
      yield* Effect.yieldNow;

      singleCapturedRequest<{ readonly type: "ping" }>(harness.fake);
      harness.fake.dispatchMessageError({ malformed: true });
      yield* Effect.yieldNow;

      const result = yield* Fiber.join(resultFiber);
      const failure = expectFailure(
        result,
        "TransientClientError",
        "worker_messageerror",
      );
      expect(failure.requestId).toBeNull();

      const lateResult = yield* Effect.result(ping(harness.transport));
      const lateFailure = expectFailure(
        lateResult,
        "TransientClientError",
        "worker_messageerror",
      );
      expect(lateFailure.requestId).toBeNull();
    }),
  );
});

it.effect("fails pending work with worker_disposed when the client scope closes", () =>
  Effect.gen(function*() {
    const scope = yield* Scope.make();
    const context = yield* Layer.buildWithScope(makeTransportHarnessLayer(), scope);
    const harness = Context.get(context, TransportHarness);

    const resultFiber = yield* Effect.result(ping(harness.transport)).pipe(
      Effect.forkChild({ startImmediately: true }),
    );
    yield* Effect.yieldNow;
    yield* waitForWorkerStart(harness.fake);
    harness.fake.dispatchReady();
    yield* Effect.yieldNow;

    singleCapturedRequest<{ readonly type: "ping" }>(harness.fake);
    yield* Scope.close(scope, Exit.void);

    const result = yield* Fiber.join(resultFiber);
    const failure = expectFailure(result, "TransientClientError", "worker_disposed");
    expect(failure.requestId).toBeNull();
    expect(harness.fake.capturedOutbound()).toContainEqual([1]);

    const lateResult = yield* Effect.result(ping(harness.transport));
    const lateFailure = expectFailure(
      lateResult,
      "TransientClientError",
      "worker_disposed",
    );
    expect(lateFailure.requestId).toBeNull();
  }),
);
