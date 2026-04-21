import { Context, Effect, Layer, Scope } from "effect";
import * as ort from "onnxruntime-web";

import {
  type WorkerRuntimeError,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import { measureDurationMs } from "./effect-timing.js";
import { EncoderModelAssets } from "./encoder-model-assets.js";
import {
  EncoderInitEventSink,
  EncoderRuntimeConfig,
} from "./encoder-runtime-config.js";
import type { FixtureTokenizer } from "./fixture-tokenizer.js";
import {
  buildWarmupFeeds,
  deriveEncoderCapabilities,
  deriveEncoderEffectiveEncodePlan,
  type EncoderEffectiveEncodePlan,
} from "./encoder-session-engine.js";
import type { EncoderCapabilities, EncoderCreateInput, EncoderHealth } from "./types.js";

export interface EncoderInferenceEngineApi {
  readonly capabilities: EncoderCapabilities;
  readonly plan: EncoderEffectiveEncodePlan;
  readonly tokenizer: FixtureTokenizer;
  readonly run: (
    feeds: ort.InferenceSession.FeedsType,
  ) => Effect.Effect<ort.InferenceSession.ReturnType, WorkerRuntimeError>;
  readonly health: () => Effect.Effect<EncoderHealth, WorkerRuntimeError>;
}

export class EncoderInferenceEngine
  extends Context.Service<EncoderInferenceEngine, EncoderInferenceEngineApi>()(
    "next-plaid-browser/EncoderInferenceEngine",
  )
{
  static readonly layer = Layer.effect(EncoderInferenceEngine)(
    makeEncoderInferenceEngine(),
  );
}

function configureOrt(): Effect.Effect<void> {
  return Effect.sync(() => {
    ort.env.wasm.wasmPaths = new URL("./ort/", self.location.href).toString();
    ort.env.wasm.proxy = false;
    ort.env.wasm.numThreads = 1;
  });
}

function validateEncodePlan(
  input: EncoderCreateInput,
  plan: EncoderEffectiveEncodePlan,
): Effect.Effect<void, WorkerRuntimeError> {
  return Effect.gen(function*() {
    if (!Number.isInteger(plan.query_length) || plan.query_length <= 0) {
      return yield* workerRuntimeError({
        operation: "encoder_inference_engine.validate_query_length",
        message: "onnx config query_length must be a positive integer",
        details: plan.query_length,
      });
    }
    if (!Number.isInteger(plan.embedding_dim) || plan.embedding_dim <= 0) {
      return yield* workerRuntimeError({
        operation: "encoder_inference_engine.validate_embedding_dim",
        message: "onnx config embedding_dim must be a positive integer",
        details: plan.embedding_dim,
      });
    }
    if (input.encoder.embedding_dim !== plan.embedding_dim) {
      return yield* workerRuntimeError({
        operation: "encoder_inference_engine.validate_encoder_identity",
        message: "encoder identity embedding_dim does not match onnx config",
        details: {
          encoderIdentity: input.encoder.embedding_dim,
          onnxConfig: plan.embedding_dim,
        },
      });
    }
  });
}

function makeEncoderInferenceEngine(): Effect.Effect<
  EncoderInferenceEngineApi,
  WorkerRuntimeError,
  EncoderInitEventSink | EncoderModelAssets | EncoderRuntimeConfig | Scope.Scope
> {
  return Effect.gen(function*() {
    const { input } = yield* EncoderRuntimeConfig;
    const eventSink = yield* EncoderInitEventSink;
    const assets = yield* EncoderModelAssets;

    yield* configureOrt();

    const plan = deriveEncoderEffectiveEncodePlan(input, assets.config);
    yield* validateEncodePlan(input, plan);
    const capabilities = deriveEncoderCapabilities(
      plan,
      assets.persistentStorage,
    );

    yield* eventSink.emit({ stage: "session_create_start" });
    let disposed = false;
    const [session, sessionCreateDurationMs] = yield* measureDurationMs(
      Effect.acquireRelease(
        Effect.tryPromise({
          try: () =>
            ort.InferenceSession.create(assets.modelBytes, {
              executionProviders: ["wasm"],
              graphOptimizationLevel: "all",
            }),
          catch: (error) =>
            workerRuntimeErrorFromUnknown(
              "encoder_inference_engine.create_session",
              error,
              "failed to create encoder session",
            ),
        }),
        (session, _exit) =>
          Effect.sync(() => {
            disposed = true;
          }).pipe(
            Effect.andThen(
              Effect.tryPromise({
                try: () => session.release(),
                catch: (error) =>
                  workerRuntimeErrorFromUnknown(
                    "encoder_inference_engine.release_session",
                    error,
                    "failed to release encoder session",
                  ),
              }),
            ),
            Effect.catchTag("WorkerRuntimeError", (error) =>
              Effect.logError(
                `encoder inference engine release failed: ${error.message}`,
              ),
            ),
          ),
      ),
    );
    yield* eventSink.emit({
      stage: "session_create_complete",
      durationMs: sessionCreateDurationMs,
    });

    yield* eventSink.emit({ stage: "warmup_start" });
    const warmupFeeds = yield* Effect.try({
      try: () => buildWarmupFeeds(plan),
      catch: (error) =>
        workerRuntimeErrorFromUnknown(
          "encoder_inference_engine.build_warmup",
          error,
          "failed to build warmup feeds",
        ),
    });
    const [, warmupDurationMs] = yield* measureDurationMs(
      Effect.tryPromise({
        try: () => session.run(warmupFeeds),
        catch: (error) =>
          workerRuntimeErrorFromUnknown(
            "encoder_inference_engine.warmup",
            error,
            "failed to warm up encoder session",
          ),
      }),
    );
    yield* eventSink.emit({
      stage: "warmup_complete",
      durationMs: warmupDurationMs,
    });

    yield* eventSink.emit({ stage: "ready", capabilities });
    const run = Effect.fn("EncoderInferenceEngine.run")(
      function*(feeds: ort.InferenceSession.FeedsType) {
        if (disposed) {
          return yield* workerRuntimeError({
            operation: "encoder_inference_engine.run",
            message: "encoder inference engine is disposed",
            details: null,
          });
        }

        return yield* Effect.tryPromise({
          try: () => session.run(feeds),
          catch: (error) =>
            workerRuntimeErrorFromUnknown(
              "encoder_inference_engine.run",
              error,
              "encoder inference failed",
            ),
        });
      },
    );
    const health = Effect.fn("EncoderInferenceEngine.health")(function*() {
      return disposed ? "degraded" : "ok";
    });

    return EncoderInferenceEngine.of({
      capabilities,
      plan,
      tokenizer: assets.tokenizer,
      run,
      health,
    });
  });
}
