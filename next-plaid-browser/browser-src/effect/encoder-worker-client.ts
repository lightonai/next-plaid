import {
  Context,
  Deferred,
  Duration,
  Effect,
  Layer,
  PubSub,
  Scope,
  Stream,
  SubscriptionRef,
  SynchronizedRef,
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
  decodeEncodeResponseSchema,
  decodeEncoderInitEventSchema,
  decodeEncoderInitResponseSchema,
} from "../model-worker/encoder-contract.js";
import {
  decodeEncoderQueryCacheKey,
  encodeEncoderInitBindingKey,
  encodeEncoderQueryCacheKey,
} from "./encoder-worker-client-schema.js";
import {
  type EncoderClientError,
  permanentClientError,
} from "./client-errors.js";
import { EncoderCacheService } from "./encoder-cache-service.js";
import { makeWorkerTransport } from "./worker-transport.js";

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
  readonly queryCacheCapacity?: number | undefined;
}

interface EncoderLifecycleControl {
  readonly initKey: string | null;
  readonly initDeferred: Deferred.Deferred<EncoderCapabilities, EncoderClientError> | null;
  readonly readyGate: Deferred.Deferred<void, EncoderClientError> | null;
}

type EncoderInitControl =
  | {
    readonly _tag: "Join";
    readonly deferred: Deferred.Deferred<EncoderCapabilities, EncoderClientError>;
  }
  | {
    readonly _tag: "Start";
    readonly deferred: Deferred.Deferred<EncoderCapabilities, EncoderClientError>;
    readonly readyGate: Deferred.Deferred<void, EncoderClientError>;
  };

const emptyLifecycleControl: EncoderLifecycleControl = {
  initKey: null,
  initDeferred: null,
  readyGate: null,
};

function publishEvent(
  pubsub: PubSub.PubSub<EncoderLifecycleEvent>,
  event: EncoderLifecycleEvent,
): Effect.Effect<void> {
  return PubSub.publish(pubsub, event).pipe(Effect.asVoid);
}

function disposedEncoderError(operation: string): EncoderClientError {
  return permanentClientError({
    cause: "encoder_disposed",
    message: "encoder worker client scope is closed",
    operation,
    details: null,
  });
}

function decodeEncoderInitResponse(
  value: unknown,
): Effect.Effect<EncoderInitResponse, EncoderClientError> {
  return decodeEncoderInitResponseSchema(value).pipe(
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
  return decodeEncodeResponseSchema(value).pipe(
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
  return decodeEncoderInitEventSchema(value).pipe(
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
      const state = yield* SubscriptionRef.make<EncoderStateSnapshot>({
        status: "empty",
        capabilities: null,
        lastError: null,
      });
      const eventPubSub = yield* PubSub.unbounded<EncoderLifecycleEvent>({
        replay: 16,
      });
      const controlState = yield* SynchronizedRef.make<EncoderLifecycleControl>(
        emptyLifecycleControl,
      );
      let clearQueryCache: Effect.Effect<void> = Effect.void;

      const failDeferred = <A>(
        deferred: Deferred.Deferred<A, EncoderClientError> | null,
        error: EncoderClientError,
      ): Effect.Effect<void> =>
        deferred === null
          ? Effect.void
          : Deferred.fail(deferred, error).pipe(Effect.asVoid);

      const clearLifecycleWaiters = (
        error: EncoderClientError,
      ): Effect.Effect<void> =>
        SynchronizedRef.modifyEffect(controlState, (control) =>
          failDeferred(control.readyGate, error).pipe(
            Effect.andThen(failDeferred(control.initDeferred, error)),
            Effect.as([undefined, emptyLifecycleControl] as const),
          ),
        );

      const getReadyGate = (): Effect.Effect<
        Deferred.Deferred<void, EncoderClientError>,
        EncoderClientError
      > =>
        SynchronizedRef.modifyEffect(controlState, (control) =>
          control.readyGate === null
            ? Effect.fail(
              permanentClientError({
                cause: "encoder_not_initialized",
                message: "encoder has not been initialized",
                operation: "encoder_worker.encode",
                details: null,
              }),
            )
            : Effect.succeed([
              control.readyGate,
              control,
            ] as const),
        );

      const transitionTerminal = Effect.fn(
        "EncoderWorkerClient.transitionTerminal",
      )(
        (options_: {
          readonly nextState: EncoderStateSnapshot;
          readonly lifecycleError: EncoderClientError;
          readonly event: EncoderLifecycleEvent;
          readonly logMessage?: string | undefined;
        }) =>
          Effect.gen(function*() {
            const snapshot = yield* SubscriptionRef.get(state);
            if (
              snapshot.status === "failed" ||
              snapshot.status === "disposed"
            ) {
              return;
            }

            yield* SubscriptionRef.set(state, options_.nextState);
            yield* clearLifecycleWaiters(options_.lifecycleError);
            yield* publishEvent(eventPubSub, options_.event);
            if (options_.logMessage !== undefined) {
              yield* Effect.logError(options_.logMessage);
            }
          }).pipe(Effect.asVoid),
      );

      const failEncoder = Effect.fn("EncoderWorkerClient.failEncoder")(
        (
          operation: string,
          error: EncoderClientError,
          capabilities: EncoderCapabilities | null,
        ) =>
          transitionTerminal({
            nextState: {
              status: "failed",
              capabilities,
              lastError: error,
            },
            lifecycleError: error,
            event: { stage: "failed", error },
            logMessage: `${operation} failed: ${error.message}`,
          }),
      );

      const transport = yield* makeWorkerTransport<EncoderWorkerRequest>({
        workerKind: "encoder",
        requestTimeout: options.requestTimeout,
        onWorkerFailure: (error) =>
          Effect.gen(function*() {
            const snapshot = yield* SubscriptionRef.get(state);
            const capabilities =
              snapshot.status === "ready" ? snapshot.capabilities : null;
            yield* clearQueryCache;
            yield* failEncoder(
              "encoder_worker.transport",
              error,
              capabilities,
            );
          }).pipe(Effect.asVoid),
      });

      const requestEncodedQuery = Effect.fn(
        "EncoderWorkerClient.requestEncodedQuery",
      )(
        (cacheKey: string) =>
          decodeEncoderQueryCacheKey(cacheKey).pipe(
            Effect.mapError((error) =>
                permanentClientError({
                  cause: "invalid_cache_key",
                  message: "failed to decode encoder query cache key",
                  operation: "encoder_worker.encode",
                  details: error,
                }),
            ),
            Effect.flatMap(({ text }) =>
              transport.request(
                { type: "encode", payload: { text } },
                {
                  operation: "encoder_worker.encode",
                  requestType: "encode",
                  decodeResponse: decodeEncodedResponse,
                },
              )
            ),
            Effect.map((response) => response.encoded),
          ),
      );

      const encoderCacheContext = yield* Layer.buildWithScope(
        EncoderCacheService.layer({
          lookup: requestEncodedQuery,
          capacity: options.queryCacheCapacity,
        }),
        clientScope,
      );
      const encoderCache = Context.get(
        encoderCacheContext,
        EncoderCacheService,
      );
      clearQueryCache = encoderCache.clear();

      yield* Effect.addFinalizer(() =>
        Effect.gen(function*() {
          yield* encoderCache.clear();
          const error = disposedEncoderError("encoder_worker.dispose");
          yield* transitionTerminal({
            nextState: {
              status: "disposed",
              capabilities: null,
              lastError: null,
            },
            lifecycleError: error,
            event: { stage: "disposed" },
          });
          yield* PubSub.shutdown(eventPubSub);
        }),
      );

      const init = Effect.fn("EncoderWorkerClient.init")(
        (input: EncoderCreateInput) =>
          Effect.gen(function*() {
            const requestedKey = yield* encodeEncoderInitBindingKey(input).pipe(
              Effect.mapError((error) =>
                permanentClientError({
                  cause: "invalid_init_key",
                  message: "failed to encode encoder init binding key",
                  operation: "encoder_worker.init",
                  details: error,
                })
              ),
            );
            const control: EncoderInitControl = yield* SynchronizedRef.modifyEffect(
              controlState,
              (
                snapshotControl,
              ): Effect.Effect<
                readonly [EncoderInitControl, EncoderLifecycleControl],
                EncoderClientError
              > =>
                Effect.gen(function*() {
                  const snapshot = yield* SubscriptionRef.get(state);
                  if (snapshot.status === "disposed") {
                    return yield* disposedEncoderError("encoder_worker.init");
                  }
                  if (snapshot.status === "failed") {
                    return yield* snapshot.lastError;
                  }

                  if (
                    snapshotControl.initKey !== null &&
                    snapshotControl.initKey !== requestedKey
                  ) {
                    return yield* permanentClientError({
                      cause: "init_requires_new_scope",
                      message:
                        "encoder client lifetime is already bound to a different encoder identity",
                      operation: "encoder_worker.init",
                      details: {
                        currentInitKey: snapshotControl.initKey,
                        requestedKey,
                      },
                    });
                  }

                  if (snapshotControl.initDeferred !== null) {
                    yield* Effect.logWarning("encoder init joined existing lifecycle");
                    return [
                      {
                        _tag: "Join",
                        deferred: snapshotControl.initDeferred,
                      },
                      snapshotControl,
                    ] as const;
                  }

                  const createdInitDeferred = yield* Deferred.make<
                    EncoderCapabilities,
                    EncoderClientError
                  >();
                  const createdReadyGate = yield* Deferred.make<void, EncoderClientError>();
                  const nextControl: EncoderLifecycleControl = {
                    initKey: requestedKey,
                    initDeferred: createdInitDeferred,
                    readyGate: createdReadyGate,
                  };

                  yield* SubscriptionRef.set(state, {
                    status: "initializing",
                    capabilities: null,
                    lastError: null,
                  });
                  yield* Effect.logInfo("encoder init started");
                  return [
                    {
                      _tag: "Start",
                      deferred: createdInitDeferred,
                      readyGate: createdReadyGate,
                    },
                    nextControl,
                  ] as const;
                }),
            );

            if (control._tag === "Start") {
              const handleInitError = (error: EncoderClientError) =>
                failEncoder("encoder_worker.init", error, null);

              const initProgram = transport.request(
                { type: "init", payload: input },
                {
                  operation: "encoder_worker.init",
                  requestType: "init",
                  decodeResponse: decodeEncoderInitResponse,
                  decodeEvent: decodeEncoderEvent,
                  onEvent: (event) => publishEvent(eventPubSub, event),
                },
              ).pipe(
                Effect.flatMap((response) =>
                  SubscriptionRef.set(state, {
                    status: "ready",
                    capabilities: response.capabilities,
                    lastError: null,
                  }).pipe(
                    Effect.tap(() => Effect.logInfo("encoder init completed")),
                    Effect.tap(() => Deferred.succeed(control.readyGate, undefined)),
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
              return yield* disposedEncoderError("encoder_worker.encode");
            }
            if (snapshot.status === "failed") {
              return yield* snapshot.lastError;
            }

            const readyGate = yield* getReadyGate();
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

            if (readySnapshot.status !== "ready") {
              return yield* permanentClientError({
                cause: "encoder_not_initialized",
                message: "encoder is not ready",
                operation: "encoder_worker.encode",
                details: { state: readySnapshot.status },
              });
            }

            const readyCapabilities = readySnapshot.capabilities;
            const handleEncodeError = (error: EncoderClientError) =>
              encoderCache.clear().pipe(
                Effect.andThen(
                  failEncoder(
                    "encoder_worker.encode",
                    error,
                    readyCapabilities,
                  ),
                ),
                Effect.andThen(Effect.fail(error)),
              );

            const cacheKey = yield* encodeEncoderQueryCacheKey(
              readyCapabilities,
              text,
            ).pipe(
              Effect.mapError((error) =>
                permanentClientError({
                  cause: "invalid_cache_key",
                  message: "failed to encode encoder query cache key",
                  operation: "encoder_worker.encode",
                  details: error,
                })
              ),
            );

            const encoded = yield* encoderCache.get(cacheKey).pipe(
              Effect.catchTags({
                TransientClientError: handleEncodeError,
                PermanentClientError: handleEncodeError,
                DegradedClientError: handleEncodeError,
              }),
              Effect.withLogSpan("encoder_worker.encode"),
              Effect.annotateLogs(logAnnotations),
            );

            return encoded;
          }),
      );

      return {
        state,
        events: Stream.fromPubSub(eventPubSub),
        init,
        encode,
      } satisfies EncoderWorkerClientApi;
    }),
    "encoder_worker.client",
  );
