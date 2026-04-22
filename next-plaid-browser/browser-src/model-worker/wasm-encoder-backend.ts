import { Clock, Context, Effect, Layer } from "effect";
import type * as ort from "onnxruntime-web";

import {
  type WorkerRuntimeError,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import { measureDurationMs } from "./effect-timing.js";
import { EncoderModelAssetCache } from "./encoder-model-asset-cache.js";
import { EncoderModelAssets } from "./encoder-model-assets.js";
import { EncoderInferenceEngine } from "./encoder-inference-engine.js";
import type { TokenizedEncoderInput } from "./fixture-tokenizer.js";
import {
  EncoderInitEventSink,
  EncoderRuntimeConfig,
} from "./encoder-runtime-config.js";
import {
  buildFeeds,
  buildTiming,
  selectOutputTensor,
  toEmbeddingRows,
} from "./encoder-session-engine.js";
import type {
  EncoderBackend,
  EncoderCreateInput,
  EncodedDocument,
  EncodedQuery,
  EncoderInitEvent,
} from "./types.js";

export class WasmEncoderBackend
  extends Context.Service<WasmEncoderBackend, EncoderBackend>()(
    "next-plaid-browser/WasmEncoderBackend",
  )
{
  static readonly layer = Layer.effect(WasmEncoderBackend)(
    makeWasmEncoderBackend(),
  );
}

export const makeWasmEncoderBackendLayer = (
  input: EncoderCreateInput,
  emitEvent: (event: EncoderInitEvent) => Effect.Effect<void>,
): Layer.Layer<WasmEncoderBackend, WorkerRuntimeError> => {
  const runtimeConfigLayer = EncoderRuntimeConfig.layer(input);
  const eventSinkLayer = EncoderInitEventSink.layer(emitEvent);
  const bootstrapDependenciesLayer = Layer.mergeAll(
    runtimeConfigLayer,
    eventSinkLayer,
  );
  const modelAssetCacheLayer = EncoderModelAssetCache.layer;
  const modelAssetsLayer = EncoderModelAssets.layer.pipe(
    Layer.provide(Layer.mergeAll(bootstrapDependenciesLayer, modelAssetCacheLayer)),
  );
  const inferenceEngineDependenciesLayer = Layer.mergeAll(
    runtimeConfigLayer,
    eventSinkLayer,
    modelAssetsLayer,
  );
  const inferenceEngineLayer = EncoderInferenceEngine.layer.pipe(
    Layer.provide(inferenceEngineDependenciesLayer),
  );

  return WasmEncoderBackend.layer.pipe(Layer.provide(inferenceEngineLayer));
};
function decodeOutput(
  results: ort.InferenceSession.ReturnType,
  expectedRows: number,
  expectedEmbeddingDim: number,
): Effect.Effect<number[][], WorkerRuntimeError> {
  return Effect.gen(function*() {
    const output = yield* Effect.try({
      try: () => selectOutputTensor(results),
      catch: (error) =>
        workerRuntimeErrorFromUnknown(
          "wasm_encoder_backend.select_output",
          error,
          "failed to resolve encoder output tensor",
        ),
    });
    if (output.type !== "float32") {
      return yield* workerRuntimeError({
        operation: "wasm_encoder_backend.validate_output_type",
        message: `expected float32 encoder output, got ${output.type}`,
        details: output.type,
      });
    }

    const tensorData = yield* Effect.tryPromise({
      try: () => output.getData(),
      catch: (error) =>
        workerRuntimeErrorFromUnknown(
          "wasm_encoder_backend.read_output",
          error,
          "failed to read encoder output",
        ),
    });
    if (!(tensorData instanceof Float32Array)) {
      return yield* workerRuntimeError({
        operation: "wasm_encoder_backend.validate_output_buffer",
        message: "expected float32 encoder output buffer",
        details: tensorData,
      });
    }
    if (output.dims.length !== 3) {
      return yield* workerRuntimeError({
        operation: "wasm_encoder_backend.validate_output_rank",
        message: `expected 3D encoder output, got dims ${output.dims.join("x")}`,
        details: output.dims,
      });
    }

    const [, rows, dim] = output.dims;
    if (rows !== expectedRows || dim !== expectedEmbeddingDim) {
      return yield* workerRuntimeError({
        operation: "wasm_encoder_backend.validate_output_shape",
        message:
          `unexpected encoder output shape ${output.dims.join("x")} expected 1x${expectedRows}x${expectedEmbeddingDim}`,
        details: {
          actual: output.dims,
          expected: [1, expectedRows, expectedEmbeddingDim],
        },
      });
    }

    return yield* Effect.try({
      try: () => toEmbeddingRows(tensorData, rows, dim),
      catch: (error) =>
        workerRuntimeErrorFromUnknown(
          "wasm_encoder_backend.normalize_output",
          error,
          "failed to normalize encoder output tensor",
        ),
    });
  });
}

function buildEncodedQuery(
  tokenized: {
    readonly inputIdValues: number[];
    readonly attentionMaskValues: number[];
  },
  embeddings: number[][],
  encoder: EncoderCreateInput["encoder"],
  queryLength: number,
  embeddingDim: number,
  total_ms: number,
  tokenize_ms: number,
  inference_ms: number,
): EncodedQuery {
  return {
    payload: {
      embeddings,
      encoder: {
        encoder_id: encoder.encoder_id,
        encoder_build: encoder.encoder_build,
        embedding_dim: embeddingDim,
        normalized: encoder.normalized,
      },
      dtype: "f32_le",
      layout: "padded_query_length",
    },
    timing: buildTiming(
      total_ms,
      tokenize_ms,
      inference_ms,
    ),
    input_ids: tokenized.inputIdValues.slice(0, queryLength),
    attention_mask: tokenized.attentionMaskValues.slice(0, queryLength),
  };
}

function buildEncodedDocument(
  tokenized: {
    readonly inputIdValues: number[];
    readonly attentionMaskValues: number[];
  },
  embeddings: number[][],
  total_ms: number,
  tokenize_ms: number,
  inference_ms: number,
): EncodedDocument {
  const rowCount = tokenized.attentionMaskValues.reduce(
    (total, value) => total + (value > 0 ? 1 : 0),
    0,
  );
  const activeEmbeddings = embeddings.slice(0, rowCount);

  return {
    payload: {
      values: activeEmbeddings.flat(),
      rows: rowCount,
      dim: activeEmbeddings[0]?.length ?? 0,
    },
    timing: buildTiming(
      total_ms,
      tokenize_ms,
      inference_ms,
    ),
    input_ids: tokenized.inputIdValues.slice(0, rowCount),
    attention_mask: tokenized.attentionMaskValues.slice(0, rowCount),
  };
}

function makeWasmEncoderBackend(): Effect.Effect<
  EncoderBackend,
  WorkerRuntimeError,
  EncoderInferenceEngine
> {
  return Effect.gen(function*() {
    const engine = yield* EncoderInferenceEngine;

    const runSequence = (
      tokenized: TokenizedEncoderInput,
      options: {
        readonly sequenceLength: number;
        readonly operation: string;
      },
    ) =>
      Effect.gen(function*() {
        const feeds = yield* Effect.try({
          try: () =>
            buildFeeds(
              tokenized,
              options.sequenceLength,
              engine.plan.uses_token_type_ids,
            ),
          catch: (error) =>
            workerRuntimeErrorFromUnknown(
              `${options.operation}.build_feeds`,
              error,
              "failed to build encoder feeds",
            ),
        });

        const [results, inference_ms] = yield* measureDurationMs(
          engine.run(feeds),
        );
        const embeddings = yield* decodeOutput(
          results,
          options.sequenceLength,
          engine.plan.embedding_dim,
        );
        return { embeddings, inference_ms };
      });

    const encodeQuery = Effect.fn("WasmEncoderBackend.encodeQuery")(function*(text: string) {
      const totalStartedAt = yield* Clock.currentTimeNanos;
      const [tokenized, tokenize_ms] = yield* measureDurationMs(
        Effect.try({
          try: () => engine.tokenizer.encodeQuery(text, engine.plan),
          catch: (error) =>
            workerRuntimeErrorFromUnknown(
              "wasm_encoder_backend.tokenize",
              error,
              "failed to tokenize encoder input",
            ),
        }),
      );

      const { embeddings, inference_ms } = yield* runSequence(
        tokenized,
        {
          sequenceLength: engine.plan.query_length,
          operation: "wasm_encoder_backend.encode_query",
        },
      );
      const total_ms =
        Number((yield* Clock.currentTimeNanos) - totalStartedAt) / 1_000_000;

      return buildEncodedQuery(
        tokenized,
        embeddings,
        engine.plan.encoder,
        engine.plan.query_length,
        engine.plan.embedding_dim,
        total_ms,
        tokenize_ms,
        inference_ms,
      );
    });

    const encodeDocument = Effect.fn("WasmEncoderBackend.encodeDocument")(function*(text: string) {
      const totalStartedAt = yield* Clock.currentTimeNanos;
      const [tokenized, tokenize_ms] = yield* measureDurationMs(
        Effect.try({
          try: () => engine.tokenizer.encodeDocument(text, engine.plan),
          catch: (error) =>
            workerRuntimeErrorFromUnknown(
              "wasm_encoder_backend.tokenize_document",
              error,
              "failed to tokenize encoder document input",
            ),
        }),
      );

      const { embeddings, inference_ms } = yield* runSequence(
        tokenized,
        {
          sequenceLength: engine.plan.document_length,
          operation: "wasm_encoder_backend.encode_document",
        },
      );
      const total_ms =
        Number((yield* Clock.currentTimeNanos) - totalStartedAt) / 1_000_000;

      return buildEncodedDocument(
        tokenized,
        embeddings,
        total_ms,
        tokenize_ms,
        inference_ms,
      );
    });

    const health = Effect.fn("WasmEncoderBackend.health")(function*() {
      return yield* engine.health();
    });

    return WasmEncoderBackend.of({
      capabilities: engine.capabilities,
      encodeQuery,
      encodeDocument,
      health,
    });
  });
}
