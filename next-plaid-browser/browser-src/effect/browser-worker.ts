import { Deferred, Effect, Layer, Scope } from "effect";
import * as Worker from "effect/unstable/workers/Worker";
import {
  WorkerError,
  WorkerReceiveError,
} from "effect/unstable/workers/WorkerError";

export const layer = (
  spawn: (id: number) => globalThis.Worker | SharedWorker | MessagePort,
): Layer.Layer<Worker.WorkerPlatform | Worker.Spawner> =>
  Layer.merge(
    layerPlatform,
    Worker.layerSpawner(spawn),
  );

export const layerPlatform: Layer.Layer<Worker.WorkerPlatform> = Layer.succeed(
  Worker.WorkerPlatform,
  Worker.makePlatform<globalThis.SharedWorker | globalThis.Worker | MessagePort>()({
    setup({ scope, worker }) {
      const port = "port" in worker ? worker.port : worker;
      return Effect.as(
        Scope.addFinalizer(
          scope,
          Effect.sync(() => {
            port.postMessage([1]);
          }),
        ),
        port,
      );
    },
    listen({ deferred, emit, port, scope }) {
      function onMessage(event: MessageEvent) {
        emit(event.data);
      }

      function failReceive(message: string, cause: unknown): void {
        Deferred.doneUnsafe(
          deferred,
          new WorkerError({
            reason: new WorkerReceiveError({
              message,
              cause,
            }),
          }).asEffect(),
        );
      }

      function onError(event: ErrorEvent) {
        failReceive(
          "An error event was emitted",
          event.error ?? event.message,
        );
      }

      function onMessageError(event: MessageEvent) {
        failReceive(
          "A messageerror event was emitted",
          event.data,
        );
      }

      port.addEventListener("message", onMessage as EventListener);
      port.addEventListener("error", onError as EventListener);
      port.addEventListener("messageerror", onMessageError as EventListener);
      if ("start" in port) {
        port.start();
      }
      return Scope.addFinalizer(
        scope,
        Effect.sync(() => {
          port.removeEventListener("message", onMessage as EventListener);
          port.removeEventListener("error", onError as EventListener);
          port.removeEventListener("messageerror", onMessageError as EventListener);
        }),
      );
    },
  }),
);
