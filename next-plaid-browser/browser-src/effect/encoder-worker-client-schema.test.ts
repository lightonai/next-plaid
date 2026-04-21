import { expect, it } from "@effect/vitest";
import { Effect } from "effect";

import type {
  EncoderCapabilities,
  EncoderCreateInput,
} from "../model-worker/types.js";
import {
  decodeEncoderQueryCacheKey,
  encodeEncoderInitBindingKey,
  encodeEncoderQueryCacheKey,
  isEncoderQueryCacheKey,
} from "./encoder-worker-client-schema.js";

function encoderInput(
  overrides: Partial<EncoderCreateInput> = {},
): EncoderCreateInput {
  return {
    encoder: {
      encoder_id: "proof-encoder",
      encoder_build: "build-1",
      embedding_dim: 4,
      normalized: true,
    },
    modelUrl: "/proof/model.onnx",
    onnxConfigUrl: "/proof/onnx_config.json",
    tokenizerUrl: "/proof/tokenizer.json",
    ...overrides,
  };
}

function encoderCapabilities(): EncoderCapabilities {
  return {
    backend: "wasm",
    threaded: false,
    persistentStorage: true,
    encoderId: "proof-encoder",
    encoderBuild: "build-1",
    embeddingDim: 4,
    queryLength: 8,
    doQueryExpansion: false,
    normalized: true,
  };
}

it.effect("encodes the init binding key deterministically across identical inputs", () =>
  Effect.gen(function*() {
    const first = yield* encodeEncoderInitBindingKey(encoderInput());
    const second = yield* encodeEncoderInitBindingKey(encoderInput());

    expect(first).toBe(second);
  }),
);

it.effect("round-trips the encoder query cache key through the shared schema", () =>
  Effect.gen(function*() {
    const encoded = yield* encodeEncoderQueryCacheKey(
      encoderCapabilities(),
      "alpha beta",
    );
    const decoded = yield* decodeEncoderQueryCacheKey(encoded);

    expect(decoded).toEqual({
      encoderId: "proof-encoder",
      encoderBuild: "build-1",
      text: "alpha beta",
    });
    expect(isEncoderQueryCacheKey(decoded)).toBe(true);
  }),
);

it.effect("rejects malformed cache-key json", () =>
  Effect.gen(function*() {
    const result = yield* Effect.result(
      decodeEncoderQueryCacheKey(`{"encoderId":"proof","encoderBuild":"build-1"}`),
    );

    expect(result._tag).toBe("Failure");
  }),
);
