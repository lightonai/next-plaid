import { Deferred, Duration, Effect, Schema, Scope, Semaphore } from "effect";
import * as Worker from "effect/unstable/workers/Worker";

import {
  type EncoderClientError,
  degradedClientError,
  permanentClientError,
  transientClientError,
  type SearchClientError,
} from "./client-errors.js";

const SuccessEnvelopeSchema = Schema.Struct({
  requestId: Schema.String,
  ok: Schema.Literal(true),
  response: Schema.Unknown,
});

const EventEnvelopeSchema = Schema.Struct({
  requestId: Schema.String,
  ok: Schema.Literal(true),
  event: Schema.Unknown,
});

const FailureEnvelopeSchema = Schema.Struct({
  requestId: Schema.String,
  ok: Schema.Literal(false),
  error: Schema.String,
});

const EnvelopeSchema = Schema.Union([
  SuccessEnvelopeSchema,
  EventEnvelopeSchema,
  FailureEnvelopeSchema,
]);

const decodeEnvelope = Schema.decodeUnknownEffect(EnvelopeSchema);

type ClientError = SearchClientError | EncoderClientError;

export const decodePassthrough = <T>(value: unknown): Effect.Effect<T, ClientError> =>
  Effect.succeed(value as T);

interface WorkerTransportRequestOptions<TResponse, TEvent> {
  readonly operation: string;
  readonly requestType: string;
  readonly decodeResponse: (value: unknown) => Effect.Effect<TResponse, ClientError>;
  readonly decodeEvent?: ((value: unknown) => Effect.Effect<TEvent, ClientError>) | undefined;
  readonly onEvent?: ((event: TEvent) => void) | undefined;
}

export interface WorkerTransport<TRequest> {
  request: <TResponse, TEvent = never>(
    request: TRequest,
    options: WorkerTransportRequestOptions<TResponse, TEvent>,
  ) => Effect.Effect<TResponse, ClientError>;
}

export interface WorkerTransportOptions {
  readonly workerKind: "search" | "encoder";
  readonly requestTimeout?: Duration.Input | undefined;
  readonly onWorkerFailure?:
    | ((error: ClientError) => Effect.Effect<void>)
    | undefined;
}

interface PendingEntry {
  readonly operation: string;
  readonly requestType: string;
  readonly decodeResponse: (value: unknown) => Effect.Effect<unknown, ClientError>;
  readonly decodeEvent: ((value: unknown) => Effect.Effect<unknown, ClientError>) | undefined;
  readonly onEvent: ((event: unknown) => void) | undefined;
  readonly deferred: Deferred.Deferred<unknown, ClientError>;
}

export const makeWorkerTransport = <TRequest>(
  options: WorkerTransportOptions,
): Effect.Effect<
  WorkerTransport<TRequest>,
  ClientError,
  Worker.WorkerPlatform | Worker.Spawner | Scope.Scope
> =>
  Effect.withLogSpan(
    Effect.gen(function*() {
      const requestTimeout = Duration.fromInputUnsafe(
        options.requestTimeout ?? Duration.seconds(15),
      );
      const semaphore = yield* Semaphore.make(1);
      const pendingRequests = new Map<string, PendingEntry>();
      const transportFailed = yield* Deferred.make<never, ClientError>();

      const platform = yield* Worker.WorkerPlatform;
      const worker = yield* platform.spawn<unknown, unknown>(0).pipe(
        Effect.mapError((error) =>
          transientClientError({
            cause: "worker_spawn_failed",
            message: error instanceof Error ? error.message : String(error),
            operation: "worker_transport.spawn",
            details: { workerKind: options.workerKind },
          }),
        ),
      );

      const failAllPending = (error: ClientError): Effect.Effect<void> =>
        Effect.sync(() => {
          for (const [, entry] of pendingRequests) {
            Deferred.doneUnsafe(entry.deferred, Effect.fail(error));
          }
          pendingRequests.clear();
        });

      const handleInbound = (message: unknown): Effect.Effect<void> =>
        Effect.gen(function*() {
          const parsed = yield* Effect.result(decodeEnvelope(message));
          if (parsed._tag === "Failure") {
            yield* Effect.logWarning(
              `worker_transport: dropping malformed envelope from ${options.workerKind}`,
            );
            return;
          }
          const envelope = parsed.success;
          const entry = pendingRequests.get(envelope.requestId);
          if (!entry) {
            yield* Effect.logDebug(
              `worker_transport: dropping envelope for unknown requestId ${envelope.requestId}`,
            );
            return;
          }

          if (envelope.ok === false) {
            pendingRequests.delete(envelope.requestId);
            yield* Deferred.fail(
              entry.deferred,
              degradedClientError({
                cause: "worker_failure_envelope",
                message: envelope.error,
                operation: entry.operation,
                requestId: envelope.requestId,
                details: { workerKind: options.workerKind },
              }),
            );
            return;
          }

          if ("event" in envelope) {
            if (!entry.decodeEvent || !entry.onEvent) {
              pendingRequests.delete(envelope.requestId);
              yield* Deferred.fail(
                entry.deferred,
                permanentClientError({
                  cause: "unexpected_event_envelope",
                  message: "worker emitted an event for an operation without event handling",
                  operation: entry.operation,
                  requestId: envelope.requestId,
                  details: { workerKind: options.workerKind },
                }),
              );
              return;
            }
            const decoded = yield* Effect.result(entry.decodeEvent(envelope.event));
            if (decoded._tag === "Failure") {
              pendingRequests.delete(envelope.requestId);
              yield* Deferred.fail(entry.deferred, decoded.failure);
              return;
            }
            entry.onEvent(decoded.success);
            return;
          }

          const decoded = yield* Effect.result(entry.decodeResponse(envelope.response));
          pendingRequests.delete(envelope.requestId);
          if (decoded._tag === "Failure") {
            yield* Deferred.fail(entry.deferred, decoded.failure);
          } else {
            yield* Deferred.succeed(entry.deferred, decoded.success);
          }
        });

      yield* Effect.forkScoped(
        worker.run(handleInbound).pipe(
          Effect.catchCause((cause) => {
            const crashed = transientClientError({
              cause: "worker_crashed",
              message: "worker run fiber terminated",
              operation: "worker_transport.run",
              details: { workerKind: options.workerKind, cause: String(cause) },
            });
            const notify = options.onWorkerFailure
              ? options.onWorkerFailure(crashed)
              : Effect.void;
            return failAllPending(crashed).pipe(
              Effect.andThen(Deferred.fail(transportFailed, crashed)),
              Effect.andThen(notify),
              Effect.andThen(
                Effect.logError(
                  `worker_transport: ${options.workerKind} worker crashed`,
                ),
              ),
            );
          }),
        ),
      );

      yield* Effect.addFinalizer(() =>
        failAllPending(
          transientClientError({
            cause: "worker_disposed",
            message: "worker transport scope closed",
            operation: "worker_transport.dispose",
            details: { workerKind: options.workerKind },
          }),
        ),
      );

      const request = Effect.fn("WorkerTransport.request")(
        <TResponse, TEvent = never>(
          payload: TRequest,
          reqOptions: WorkerTransportRequestOptions<TResponse, TEvent>,
        ) =>
          semaphore.withPermit(
            Effect.gen(function*() {
              const failed = yield* Deferred.isDone(transportFailed);
              if (failed) {
                return yield* Deferred.await(transportFailed);
              }

              const requestId = crypto.randomUUID();
              const deferred = yield* Deferred.make<unknown, ClientError>();

              pendingRequests.set(requestId, {
                operation: reqOptions.operation,
                requestType: reqOptions.requestType,
                decodeResponse: reqOptions.decodeResponse as (
                  value: unknown,
                ) => Effect.Effect<unknown, ClientError>,
                decodeEvent: reqOptions.decodeEvent as
                  | ((value: unknown) => Effect.Effect<unknown, ClientError>)
                  | undefined,
                onEvent: reqOptions.onEvent as ((event: unknown) => void) | undefined,
                deferred,
              });

              yield* worker.send({ requestId, request: payload }).pipe(
                Effect.catchCause((cause) =>
                  Effect.fail(
                    transientClientError({
                      cause: "send_failed",
                      message: "failed to send request to worker",
                      operation: reqOptions.operation,
                      requestId,
                      details: {
                        workerKind: options.workerKind,
                        cause: String(cause),
                      },
                    }),
                  ),
                ),
              );

              const result = yield* Deferred.await(deferred).pipe(
                Effect.timeoutOrElse({
                  duration: requestTimeout,
                  orElse: () =>
                    Effect.fail(
                      transientClientError({
                        cause: "timeout",
                        message: `worker request timed out after ${Duration.toMillis(requestTimeout)}ms`,
                        operation: reqOptions.operation,
                        requestId,
                        details: {
                          requestType: reqOptions.requestType,
                          workerKind: options.workerKind,
                        },
                      }),
                    ),
                }),
                Effect.onExit(() =>
                  Effect.sync(() => {
                    pendingRequests.delete(requestId);
                  }),
                ),
              );

              return result as TResponse;
            }).pipe(
              Effect.annotateLogs({
                worker_kind: options.workerKind,
                operation: reqOptions.operation,
                request_type: reqOptions.requestType,
              }),
              Effect.withLogSpan("worker_transport.request"),
            ),
          ),
      );

      return {
        request,
      } satisfies WorkerTransport<TRequest>;
    }),
    "worker_transport",
  );
