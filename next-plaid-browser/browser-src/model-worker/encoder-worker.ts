import * as BrowserWorkerRunner from "@effect/platform-browser/BrowserWorkerRunner";
import { Cause, Effect, Layer } from "effect";

import {
  makeWorkerEntrypointLayer,
} from "../effect/worker-entrypoint.js";
import { decodeEncoderWorkerRequest } from "./encoder-contract.js";
import {
  EncoderRuntimeCoordinator,
  EncoderRuntimeCoordinatorLive,
} from "./encoder-runtime-coordinator.js";
import type {
  EncoderInitEvent,
  EncoderWorkerRequest,
  EncoderWorkerResponse,
} from "./types.js";

const EncoderWorkerLive = makeWorkerEntrypointLayer<
  EncoderWorkerRequest,
  EncoderWorkerResponse,
  EncoderInitEvent,
  EncoderRuntimeCoordinator
>({
  workerKind: "encoder",
  decodeRequest: decodeEncoderWorkerRequest,
  handleRequest: (_requestId, request, emitEvent) =>
    Effect.gen(function*() {
      const coordinator = yield* EncoderRuntimeCoordinator;
      return yield* coordinator.handleRequest(request, emitEvent);
    }),
}).pipe(
  Layer.provide(EncoderRuntimeCoordinatorLive),
  Layer.provide(BrowserWorkerRunner.layer),
);

Effect.runFork(
  Layer.launch(EncoderWorkerLive).pipe(
    Effect.tapCause((cause) =>
      Effect.logError(
        `encoder worker failed during launch:\n${Cause.pretty(cause)}`,
      ),
    ),
  ),
);
