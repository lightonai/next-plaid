import {
  Context,
  Deferred,
  Duration,
  Effect,
  Layer,
  Queue,
  Schema,
  Scope,
  Semaphore,
  Stream,
  SubscriptionRef,
} from "effect";
import * as Worker from "effect/unstable/workers/Worker";

import type {
  EncodedQuery,
  EncoderCapabilities,
  EncoderCreateInput,
  EncoderInitEvent,
  EncoderInitResponse,
  EncodeResponse,
  EncoderWorkerRequest,
} from "../model-worker/types.js";
import {
  type EncoderClientError,
  permanentClientError,
} from "./client-errors.js";
import { makeWorkerTransport } from "./worker-transport.js";

const EncoderIdentitySchema = Schema.Struct({
  encoder_id: Schema.String,
  encoder_build: Schema.String,
  embedding_dim: Schema.Number,
  normalized: Schema.Boolean,
});

const EncoderCapabilitiesSchema = Schema.Struct({
  backend: Schema.Literal("wasm"),
  threaded: Schema.Boolean,
  persistentStorage: Schema.Boolean,
  encoderId: Schema.String,
  encoderBuild: Schema.String,
  embeddingDim: Schema.Number,
  queryLength: Schema.Number,
  doQueryExpansion: Schema.Boolean,
  normalized: Schema.Boolean,
});

const EncodeTimingBreakdownSchema = Schema.Struct({
  total_ms: Schema.Number,
  tokenize_ms: Schema.Number,
  inference_ms: Schema.Number,
});

const EncodedQuerySchema = Schema.Struct({
  payload: Schema.Struct({
    embeddings: Schema.Array(Schema.Array(Schema.Number)),
    encoder: EncoderIdentitySchema,
    dtype: Schema.Literal("f32_le"),
    layout: Schema.Union([Schema.Literal("ragged"), Schema.Literal("padded_query_length")]),
  }),
  timing: EncodeTimingBreakdownSchema,
  input_ids: Schema.Array(Schema.Number),
  attention_mask: Schema.Array(Schema.Number),
});

const EncoderInitResponseSchema = Schema.Struct({
  type: Schema.Literal("encoder_ready"),
  state: Schema.Literal("ready"),
  capabilities: EncoderCapabilitiesSchema,
});

const EncodeResponseSchema = Schema.Struct({
  type: Schema.Literal("encoded_query"),
  encoded: EncodedQuerySchema,
});

const EncoderInitEventSchema = Schema.Union([
  Schema.Struct({
    stage: Schema.Literal("fetch_start"),
    url: Schema.String,
    expectedBytes: Schema.NullOr(Schema.Number),
  }),
  Schema.Struct({
    stage: Schema.Literal("fetch_complete"),
    url: Schema.String,
    bytesReceived: Schema.Number,
  }),
  Schema.Struct({
    stage: Schema.Literal("session_create_start"),
  }),
  Schema.Struct({
    stage: Schema.Literal("session_create_complete"),
    durationMs: Schema.Number,
  }),
  Schema.Struct({
    stage: Schema.Literal("warmup_start"),
  }),
  Schema.Struct({
    stage: Schema.Literal("warmup_complete"),
    durationMs: Schema.Number,
  }),
  Schema.Struct({
    stage: Schema.Literal("ready"),
    capabilities: EncoderCapabilitiesSchema,
  }),
]);

export type EncoderStateSnapshot =
  | { status: "empty"; capabilities: null; lastError: null }
  | { status: "initializing"; capabilities: null; lastError: null }
  | { status: "ready"; capabilities: EncoderCapabilities; lastError: null }
  | { status: "failed"; capabilities: EncoderCapabilities | null; lastError: EncoderClientError }
  | { status: "disposed"; capabilities: null; lastError: null };

export type EncoderLifecycleEvent =
  | EncoderInitEvent
  | { stage: "failed"; error: EncoderClientError }
  | { stage: "disposed" };

export interface EncoderWorkerClientApi {
  readonly state: SubscriptionRef.SubscriptionRef<EncoderStateSnapshot>;
  readonly events: Stream.Stream<EncoderLifecycleEvent, never>;
  readonly init: (
    input: EncoderCreateInput,
  ) => Effect.Effect<EncoderCapabilities, EncoderClientError>;
  readonly encode: (
    args: { text: string; requestId?: string },
  ) => Effect.Effect<EncodedQuery, EncoderClientError>;
}

export class EncoderWorkerClient
  extends Context.Service<EncoderWorkerClient, EncoderWorkerClientApi>()(
    "next-plaid-browser/EncoderWorkerClient",
  )
{
  static layer = (
    options: EncoderWorkerClientOptions = {},
  ): Layer.Layer<
    EncoderWorkerClient,
    EncoderClientError,
    Worker.WorkerPlatform | Worker.Spawner
  > => Layer.effect(EncoderWorkerClient)(makeEncoderWorkerClient(options));
}

export interface EncoderWorkerClientOptions {
  readonly requestTimeout?: Duration.Input | undefined;
}

function encoderInputKey(input: EncoderCreateInput): string {
  return JSON.stringify({
    encoder: input.encoder,
    modelUrl: input.modelUrl,
    onnxConfigUrl: input.onnxConfigUrl,
    tokenizerUrl: input.tokenizerUrl,
    prefer: input.prefer ?? null,
  });
}

function publishEvent(
  queue: Queue.Queue<EncoderLifecycleEvent>,
  event: EncoderLifecycleEvent,
): Effect.Effect<void> {
  return Queue.offer(queue, event).pipe(Effect.asVoid);
}

function decodeEncoderInitResponse(
  value: unknown,
): Effect.Effect<EncoderInitResponse, EncoderClientError> {
  return Schema.decodeUnknownEffect(EncoderInitResponseSchema)(value).pipe(
    Effect.mapError((error) =>
      permanentClientError({
        cause: "decode_failed",
        message: `failed to decode encoder init response: ${String(error)}`,
        operation: "encoder_worker.init",
        details: error,
      }),
    ),
  );
}

function decodeEncodedResponse(
  value: unknown,
): Effect.Effect<EncodeResponse, EncoderClientError> {
  return Schema.decodeUnknownEffect(EncodeResponseSchema)(value).pipe(
    Effect.map((decoded) => ({
      type: decoded.type,
      encoded: {
        payload: {
          embeddings: decoded.encoded.payload.embeddings.map((row) => [...row]),
          encoder: decoded.encoded.payload.encoder,
          dtype: decoded.encoded.payload.dtype,
          layout: decoded.encoded.payload.layout,
        },
        timing: {
          total_ms: decoded.encoded.timing.total_ms,
          tokenize_ms: decoded.encoded.timing.tokenize_ms,
          inference_ms: decoded.encoded.timing.inference_ms,
        },
        input_ids: [...decoded.encoded.input_ids],
        attention_mask: [...decoded.encoded.attention_mask],
      },
    })),
    Effect.mapError((error) =>
      permanentClientError({
        cause: "decode_failed",
        message: `failed to decode encoder encode response: ${String(error)}`,
        operation: "encoder_worker.encode",
        details: error,
      }),
    ),
  );
}

function decodeEncoderEvent(
  value: unknown,
): Effect.Effect<EncoderInitEvent, EncoderClientError> {
  return Schema.decodeUnknownEffect(EncoderInitEventSchema)(value).pipe(
    Effect.mapError((error) =>
      permanentClientError({
        cause: "decode_failed",
        message: `failed to decode encoder init event: ${String(error)}`,
        operation: "encoder_worker.init",
        details: error,
      }),
    ),
  );
}

export const makeEncoderWorkerClient = (
  options: EncoderWorkerClientOptions = {},
): Effect.Effect<
  EncoderWorkerClientApi,
  EncoderClientError,
  Worker.WorkerPlatform | Worker.Spawner | Scope.Scope
> =>
  Effect.withLogSpan(
    Effect.gen(function*() {
      const clientScope = yield* Effect.scope;
      const state = yield* Effect.acquireRelease(
        SubscriptionRef.make<EncoderStateSnapshot>({
          status: "empty",
          capabilities: null,
          lastError: null,
        }),
        (ref) =>
          SubscriptionRef.set(ref, {
            status: "disposed",
            capabilities: null,
            lastError: null,
          }),
      );
      const eventQueue = yield* Effect.acquireRelease(
        Queue.unbounded<EncoderLifecycleEvent>(),
        (queue) =>
          publishEvent(queue, { stage: "disposed" }).pipe(
            Effect.andThen(Queue.shutdown(queue)),
            Effect.asVoid,
          ),
      );
      const lifecycleSemaphore = yield* Semaphore.make(1);
      let initDeferred: Deferred.Deferred<EncoderCapabilities, EncoderClientError> | null = null;
      let readyGate: Deferred.Deferred<void, EncoderClientError> | null = null;
      let currentInitKey: string | null = null;

      const failEncoder = Effect.fn("EncoderWorkerClient.failEncoder")(
        (
          operation: string,
          error: EncoderClientError,
          capabilities: EncoderCapabilities | null,
        ) =>
          SubscriptionRef.set(state, {
            status: "failed",
            capabilities,
            lastError: error,
          }).pipe(
            Effect.tap(() => publishEvent(eventQueue, { stage: "failed", error })),
            Effect.tap(() => Effect.logError(`${operation} failed: ${error.message}`)),
          ),
      );

      const transport = yield* makeWorkerTransport<EncoderWorkerRequest>({
        workerKind: "encoder",
        requestTimeout: options.requestTimeout,
        onWorkerFailure: (error) =>
          Effect.gen(function*() {
            const snapshot = yield* SubscriptionRef.get(state);
            if (snapshot.status !== "ready") {
              return;
            }
            yield* SubscriptionRef.set(state, {
              status: "failed",
              capabilities: snapshot.capabilities,
              lastError: error,
            });
          }).pipe(Effect.asVoid),
      });

      const init = Effect.fn("EncoderWorkerClient.init")(
        (input: EncoderCreateInput) =>
          Effect.gen(function*() {
            const requestedKey = encoderInputKey(input);
            const control = yield* lifecycleSemaphore.withPermit(
              Effect.gen(function*() {
                if (currentInitKey !== null && currentInitKey !== requestedKey) {
                  return yield* permanentClientError({
                    cause: "init_requires_new_scope",
                    message:
                      "encoder client lifetime is already bound to a different encoder identity",
                    operation: "encoder_worker.init",
                    details: { currentInitKey, requestedKey },
                  });
                }

                if (initDeferred !== null) {
                  yield* Effect.logWarning("encoder init joined existing lifecycle");
                  return {
                    deferred: initDeferred,
                    ready: readyGate!,
                    shouldStart: false as const,
                  };
                }

                const createdInitDeferred = yield* Deferred.make<
                  EncoderCapabilities,
                  EncoderClientError
                >();
                const createdReadyGate = yield* Deferred.make<void, EncoderClientError>();

                initDeferred = createdInitDeferred;
                readyGate = createdReadyGate;
                currentInitKey = requestedKey;

                yield* SubscriptionRef.set(state, {
                  status: "initializing",
                  capabilities: null,
                  lastError: null,
                });
                yield* Effect.logInfo("encoder init started");
                return {
                  deferred: createdInitDeferred,
                  ready: createdReadyGate,
                  shouldStart: true as const,
                };
              }),
            );

            if (control.shouldStart) {
              const handleInitError = (error: EncoderClientError) =>
                failEncoder("encoder_worker.init", error, null).pipe(
                  Effect.andThen(Deferred.fail(control.ready, error)),
                  Effect.andThen(Deferred.fail(control.deferred, error)),
                  Effect.asVoid,
                );

              const initProgram = transport.request(
                { type: "init", payload: input },
                {
                  operation: "encoder_worker.init",
                  requestType: "init",
                  decodeResponse: decodeEncoderInitResponse,
                  decodeEvent: decodeEncoderEvent,
                  onEvent: (event) => publishEvent(eventQueue, event),
                },
              ).pipe(
                Effect.flatMap((response) =>
                  SubscriptionRef.set(state, {
                    status: "ready",
                    capabilities: response.capabilities,
                    lastError: null,
                  }).pipe(
                    Effect.tap(() => Effect.logInfo("encoder init completed")),
                    Effect.tap(() => Deferred.succeed(control.ready, undefined)),
                    Effect.tap(() => Deferred.succeed(control.deferred, response.capabilities)),
                  ),
                ),
                Effect.catchTags({
                  TransientClientError: handleInitError,
                  PermanentClientError: handleInitError,
                  DegradedClientError: handleInitError,
                }),
                Effect.withLogSpan("encoder_worker.init"),
                Effect.annotateLogs({
                  worker_kind: "encoder",
                  operation: "encoder_worker.init",
                  encoder_id: input.encoder.encoder_id,
                  encoder_build: input.encoder.encoder_build,
                }),
              );

              yield* Effect.forkIn(initProgram, clientScope, { startImmediately: true });
            }

            return yield* Deferred.await(control.deferred);
          }),
      );

      const encode = Effect.fn("EncoderWorkerClient.encode")(
        ({ text, requestId }: { text: string; requestId?: string }) =>
          Effect.gen(function*() {
            const snapshot = yield* SubscriptionRef.get(state);
            if (snapshot.status === "empty") {
              return yield* permanentClientError({
                cause: "encoder_not_initialized",
                message: "encoder has not been initialized",
                operation: "encoder_worker.encode",
                details: null,
              });
            }
            if (snapshot.status === "disposed") {
              return yield* permanentClientError({
                cause: "encoder_disposed",
                message: "encoder worker client scope is closed",
                operation: "encoder_worker.encode",
                details: null,
              });
            }
            if (snapshot.status === "failed") {
              return yield* snapshot.lastError;
            }
            if (readyGate === null) {
              return yield* permanentClientError({
                cause: "encoder_not_initialized",
                message: "encoder has not been initialized",
                operation: "encoder_worker.encode",
                details: null,
              });
            }

            yield* Deferred.await(readyGate);
            const readySnapshot = yield* SubscriptionRef.get(state);
            const logAnnotations: Record<string, string | number> = {
              worker_kind: "encoder",
              operation: "encoder_worker.encode",
              query_char_len: text.length,
            };
            if (requestId !== undefined) {
              logAnnotations.request_id = requestId;
            }
            if (readySnapshot.status === "ready") {
              logAnnotations.encoder_id = readySnapshot.capabilities.encoderId;
            }

            const handleEncodeError = (error: EncoderClientError) =>
              failEncoder(
                "encoder_worker.encode",
                error,
                snapshot.status === "ready" ? snapshot.capabilities : null,
              ).pipe(Effect.andThen(Effect.fail(error)));

            const response = yield* transport.request(
              { type: "encode", payload: { text } },
              {
                operation: "encoder_worker.encode",
                requestType: "encode",
                decodeResponse: decodeEncodedResponse,
              },
            ).pipe(
              Effect.catchTags({
                TransientClientError: handleEncodeError,
                PermanentClientError: handleEncodeError,
                DegradedClientError: handleEncodeError,
              }),
              Effect.withLogSpan("encoder_worker.encode"),
              Effect.annotateLogs(logAnnotations),
            );

            return response.encoded;
          }),
      );

      return {
        state,
        events: Stream.fromQueue(eventQueue),
        init,
        encode,
      } satisfies EncoderWorkerClientApi;
    }),
    "encoder_worker.client",
  );
