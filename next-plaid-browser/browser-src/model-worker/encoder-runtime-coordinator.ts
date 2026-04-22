import { Cause, Context, Effect, Layer, Ref, Scope, Semaphore } from "effect";
import * as Exit from "effect/Exit";

import {
  type WorkerRuntimeError,
  workerRuntimeError,
} from "../effect/worker-runtime-errors.js";
import {
  makeWasmEncoderBackendLayer,
  WasmEncoderBackend,
} from "./wasm-encoder-backend.js";
import type {
  EncoderBackend,
  EncoderHealthResponse,
  EncoderInitEvent,
  EncoderInitResponse,
  EncoderState,
  EncoderWorkerRequest,
  EncoderWorkerResponse,
} from "./types.js";

interface EncoderRuntimeState {
  readonly backend: EncoderBackend | null;
  readonly backendScope: Scope.Closeable | null;
  readonly state: EncoderState;
  readonly lastError: string | null;
}

interface EncoderRuntimeCoordinatorApi {
  readonly handleRequest: (
    request: EncoderWorkerRequest,
    emitEvent: (event: EncoderInitEvent) => Effect.Effect<void>,
  ) => Effect.Effect<EncoderWorkerResponse, WorkerRuntimeError>;
}

export class EncoderRuntimeCoordinator
  extends Context.Service<EncoderRuntimeCoordinator, EncoderRuntimeCoordinatorApi>()(
    "next-plaid-browser/EncoderRuntimeCoordinator",
  )
{}

function renderFailure(cause: Cause.Cause<unknown>): string {
  const rendered = Cause.pretty(cause).trim();
  return rendered.length > 0 ? rendered : "encoder worker request failed";
}

export const EncoderRuntimeCoordinatorLive = Layer.effect(
  EncoderRuntimeCoordinator,
  Effect.gen(function*() {
    const requestSemaphore = yield* Semaphore.make(1);
    const stateRef = yield* Ref.make<EncoderRuntimeState>({
      backend: null,
      backendScope: null,
      state: "empty",
      lastError: null,
    });

    const setState = (
      nextState: EncoderRuntimeState,
    ): Effect.Effect<void> => Ref.set(stateRef, nextState);

    const failState = (
      cause: Cause.Cause<unknown>,
    ): Effect.Effect<void> =>
      Effect.gen(function*() {
        const snapshot = yield* Ref.get(stateRef);
        if (snapshot.backendScope !== null) {
          yield* Scope.close(snapshot.backendScope, Exit.void);
        }
        yield* Ref.set(stateRef, {
          backend: null,
          backendScope: null,
          state: "failed",
          lastError: renderFailure(cause),
        });
      });

    const handleRequest = Effect.fn("EncoderRuntimeCoordinator.handleRequest")(
      (
        request: EncoderWorkerRequest,
        emitEvent: (event: EncoderInitEvent) => Effect.Effect<void>,
      ) =>
        requestSemaphore.withPermit(
          Effect.gen(function*() {
            const responseExit = yield* Effect.exit(
              Effect.gen(function*() {
                switch (request.type) {
                  case "init": {
                    const snapshot = yield* Ref.get(stateRef);
                    if (snapshot.backendScope !== null) {
                      yield* Scope.close(snapshot.backendScope, Exit.void);
                    }

                    yield* setState({
                      backend: null,
                      backendScope: null,
                      state: "initializing",
                      lastError: null,
                    });

                    const backendScope = yield* Scope.make();
                    const backendContext = yield* Layer.buildWithScope(
                      makeWasmEncoderBackendLayer(request.payload, emitEvent),
                      backendScope,
                    ).pipe(
                      Effect.onExit((exit) =>
                        exit._tag === "Failure"
                          ? Scope.close(backendScope, exit)
                          : Effect.void,
                      ),
                    );
                    const backend = Context.get(
                      backendContext,
                      WasmEncoderBackend,
                    );
                    yield* setState({
                      backend,
                      backendScope,
                      state: "ready",
                      lastError: null,
                    });

                    const response: EncoderInitResponse = {
                      type: "encoder_ready",
                      state: "ready",
                      capabilities: backend.capabilities,
                    };
                    return response;
                  }

                  case "health": {
                    const snapshot = yield* Ref.get(stateRef);
                    const response: EncoderHealthResponse = {
                      type: "encoder_health",
                      state: snapshot.state,
                      health:
                        snapshot.backend === null
                          ? "degraded"
                          : yield* snapshot.backend.health(),
                      capabilities: snapshot.backend?.capabilities ?? null,
                      last_error: snapshot.lastError,
                    };
                    return response;
                  }

                  case "encode_query": {
                    const snapshot = yield* Ref.get(stateRef);
                    if (
                      snapshot.backend === null ||
                      snapshot.state !== "ready"
                    ) {
                      return yield* workerRuntimeError({
                        operation: "encoder_runtime_coordinator.encode_query",
                        message: "encoder worker is not ready",
                        details: { state: snapshot.state },
                      });
                    }

                    return {
                      type: "encoded_query",
                      encoded: yield* snapshot.backend.encodeQuery(
                        request.payload.text,
                      ),
                    } satisfies EncoderWorkerResponse;
                  }

                  case "encode_document": {
                    const snapshot = yield* Ref.get(stateRef);
                    if (
                      snapshot.backend === null ||
                      snapshot.state !== "ready"
                    ) {
                      return yield* workerRuntimeError({
                        operation: "encoder_runtime_coordinator.encode_document",
                        message: "encoder worker is not ready",
                        details: { state: snapshot.state },
                      });
                    }

                    return {
                      type: "encoded_document",
                      encoded: yield* snapshot.backend.encodeDocument(
                        request.payload.text,
                      ),
                    } satisfies EncoderWorkerResponse;
                  }

                  case "dispose": {
                    const snapshot = yield* Ref.get(stateRef);
                    if (snapshot.backendScope !== null) {
                      yield* Scope.close(snapshot.backendScope, Exit.void);
                    }

                    yield* setState({
                      backend: null,
                      backendScope: null,
                      state: "disposed",
                      lastError: null,
                    });

                    return {
                      type: "encoder_disposed",
                    } satisfies EncoderWorkerResponse;
                  }

                  default: {
                    const impossible: never = request;
                    return yield* workerRuntimeError({
                      operation: "encoder_runtime_coordinator.handle_request",
                      message:
                        `unsupported encoder request ${(impossible as { type?: string }).type ?? "unknown"}`,
                      details: impossible,
                    });
                  }
                }
              }),
            );

            if (responseExit._tag === "Failure") {
              yield* failState(responseExit.cause);
              return yield* workerRuntimeError({
                operation: "encoder_runtime_coordinator.handle_request",
                message: renderFailure(responseExit.cause),
                details: responseExit.cause,
              });
            }

            return responseExit.value;
          }),
        ),
    );

    return EncoderRuntimeCoordinator.of({
      handleRequest,
    });
  }),
);
