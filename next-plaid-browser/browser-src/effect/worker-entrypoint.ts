import { Cause, Effect, Layer, Schema, Scope } from "effect";
import * as WorkerRunner from "effect/unstable/workers/WorkerRunner";
import type { WorkerError } from "effect/unstable/workers/WorkerError";

import type { WorkerResponseEnvelope } from "../shared/worker-envelope.js";

const WorkerRequestEnvelopeSchema = Schema.Struct({
  requestId: Schema.String,
  request: Schema.Unknown,
});

const decodeWorkerRequestEnvelope = Schema.decodeUnknownEffect(
  WorkerRequestEnvelopeSchema,
);

function renderFailure(cause: Cause.Cause<unknown>): string {
  const rendered = Cause.pretty(cause).trim();
  return rendered.length > 0 ? rendered : "worker request failed";
}

export interface WorkerEntrypointOptions<TRequest, TResponse, TEvent, R> {
  readonly workerKind: string;
  readonly decodeRequest: (
    value: unknown,
  ) => Effect.Effect<TRequest, unknown, R>;
  readonly handleRequest: (
    requestId: string,
    request: TRequest,
    emitEvent: (event: TEvent) => Effect.Effect<void>,
  ) => Effect.Effect<TResponse, unknown, R>;
}

export const makeWorkerEntrypointLayer = <TRequest, TResponse, TEvent, R>(
  options: WorkerEntrypointOptions<TRequest, TResponse, TEvent, R>,
): Layer.Layer<
  never,
  WorkerError,
  WorkerRunner.WorkerRunnerPlatform | Exclude<R, Scope.Scope>
> =>
  Layer.effectDiscard(
    Effect.gen(function*() {
      const platform = yield* WorkerRunner.WorkerRunnerPlatform;
      const runner = yield* platform.start<
        WorkerResponseEnvelope<TResponse, TEvent>,
        unknown
      >();

      const sendFailure = (
        portId: number,
        requestId: string,
        error: string,
      ): Effect.Effect<void> =>
        runner.send(portId, {
          requestId,
          ok: false,
          error,
        });

      const sendEvent = (
        portId: number,
        requestId: string,
        event: TEvent,
      ): Effect.Effect<void> =>
        runner.send(portId, {
          requestId,
          ok: true,
          event,
        });

      const sendSuccess = (
        portId: number,
        requestId: string,
        response: TResponse,
      ): Effect.Effect<void> =>
        runner.send(portId, {
          requestId,
          ok: true,
          response,
        });

      const handleMessage = (
        portId: number,
        rawMessage: unknown,
      ): Effect.Effect<void, unknown, R> =>
        Effect.gen(function*() {
          const envelopeResult = yield* Effect.result(
            decodeWorkerRequestEnvelope(rawMessage),
          );
          if (envelopeResult._tag === "Failure") {
            yield* sendFailure(
              portId,
              "",
              `invalid ${options.workerKind} worker request envelope: ${String(envelopeResult.failure)}`,
            );
            return;
          }

          const envelope = envelopeResult.success;
          const requestResult = yield* Effect.result(
            options.decodeRequest(envelope.request),
          );
          if (requestResult._tag === "Failure") {
            yield* sendFailure(
              portId,
              envelope.requestId,
              `invalid ${options.workerKind} worker request: ${String(requestResult.failure)}`,
            );
            return;
          }

          const responseExit = yield* Effect.exit(
            options.handleRequest(
              envelope.requestId,
              requestResult.success,
              (event) => sendEvent(portId, envelope.requestId, event),
            ),
          );
          if (responseExit._tag === "Failure") {
            yield* sendFailure(
              portId,
              envelope.requestId,
              renderFailure(responseExit.cause),
            );
            return;
          }

          yield* sendSuccess(
            portId,
            envelope.requestId,
            responseExit.value,
          );
        }).pipe(
          Effect.withLogSpan(`${options.workerKind}_worker.handle_request`),
          Effect.annotateLogs({
            worker_kind: options.workerKind,
          }),
        );

      yield* Effect.forkScoped(
        runner.run(handleMessage).pipe(
          Effect.tapCause((cause) =>
            Effect.logError(
              `${options.workerKind} worker runner failed:\n${Cause.pretty(cause)}`,
            ),
          ),
        ),
      );
    }),
  );
