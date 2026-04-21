import * as ort from "onnxruntime-web";

import { workerRuntimeError } from "../effect/worker-runtime-errors.js";
import type { TokenizedQuery } from "./fixture-tokenizer.js";
import type {
  EncodeTimingBreakdown,
  EncoderCapabilities,
  EncoderCreateInput,
  OnnxConfig,
} from "./types.js";

export interface EncoderEffectiveEncodePlan extends OnnxConfig {
  readonly encoder: EncoderCreateInput["encoder"];
}

function ensureFinite(values: Iterable<number>, label: string): void {
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw workerRuntimeError({
        operation: "encoder_session_engine.ensure_finite",
        message: `${label} must not contain NaN or Infinity`,
        details: { label, value },
      });
    }
  }
}

export function selectOutputTensor(
  results: ort.InferenceSession.ReturnType,
): ort.Tensor {
  const candidate =
    "output" in results ? results.output : Object.values(results)[0];
  if (!candidate || typeof candidate !== "object" || !("dims" in candidate)) {
    throw workerRuntimeError({
      operation: "encoder_session_engine.select_output",
      message: "model inference did not produce a tensor output",
      details: results,
    });
  }
  return candidate as ort.Tensor;
}

export function toEmbeddingRows(
  data: Float32Array,
  rows: number,
  dim: number,
): number[][] {
  const embeddings: number[][] = [];
  for (let row = 0; row < rows; row += 1) {
    const start = row * dim;
    const values = Array.from(data.slice(start, start + dim));
    ensureFinite(values, "encoder output");
    embeddings.push(values);
  }
  return embeddings;
}

export function buildTiming(
  total_ms: number,
  tokenize_ms: number,
  inference_ms: number,
): EncodeTimingBreakdown {
  return {
    total_ms,
    tokenize_ms,
    inference_ms,
  };
}

export function buildFeeds(
  tokenized: TokenizedQuery,
  plan: EncoderEffectiveEncodePlan,
): ort.InferenceSession.FeedsType {
  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor(
      "int64",
      tokenized.inputIds,
      [1, plan.query_length],
    ),
    attention_mask: new ort.Tensor(
      "int64",
      tokenized.attentionMask,
      [1, plan.query_length],
    ),
  };

  if (plan.uses_token_type_ids) {
    if (tokenized.tokenTypeIds === null) {
      throw workerRuntimeError({
        operation: "encoder_session_engine.build_feeds",
        message: "token type ids are required by the encode plan",
        details: plan,
      });
    }
    feeds.token_type_ids = new ort.Tensor(
      "int64",
      tokenized.tokenTypeIds,
      [1, plan.query_length],
    );
  }

  return feeds as ort.InferenceSession.FeedsType;
}

export function buildWarmupFeeds(
  plan: EncoderEffectiveEncodePlan,
): ort.InferenceSession.FeedsType {
  const feeds: Record<string, ort.Tensor> = {
    input_ids: new ort.Tensor(
      "int64",
      new BigInt64Array(plan.query_length),
      [1, plan.query_length],
    ),
    attention_mask: new ort.Tensor(
      "int64",
      BigInt64Array.from({ length: plan.query_length }, () => 1n),
      [1, plan.query_length],
    ),
  };

  if (plan.uses_token_type_ids) {
    feeds.token_type_ids = new ort.Tensor(
      "int64",
      new BigInt64Array(plan.query_length),
      [1, plan.query_length],
    );
  }

  return feeds as ort.InferenceSession.FeedsType;
}

export function deriveEncoderEffectiveEncodePlan(
  input: EncoderCreateInput,
  config: OnnxConfig,
): EncoderEffectiveEncodePlan {
  return {
    ...config,
    do_query_expansion: false,
    encoder: input.encoder,
  };
}

export function deriveEncoderCapabilities(
  plan: EncoderEffectiveEncodePlan,
  persistentStorage: boolean,
): EncoderCapabilities {
  return {
    backend: "wasm",
    threaded: false,
    persistentStorage,
    encoderId: plan.encoder.encoder_id,
    encoderBuild: plan.encoder.encoder_build,
    embeddingDim: plan.embedding_dim,
    queryLength: plan.query_length,
    doQueryExpansion: plan.do_query_expansion,
    normalized: plan.encoder.normalized,
  };
}
