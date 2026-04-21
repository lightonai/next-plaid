import { expect, it } from "@effect/vitest";
import { Effect } from "effect";

import {
  decodeEncodeResponseSchema,
  isEncodeResponse,
  isEncoderInitEvent,
  isEncoderInitResponse,
  isEncoderWorkerRequest,
} from "./encoder-contract.js";

it.effect("exposes schema-derived guards for encoder contract shapes", () =>
  Effect.sync(() => {
    expect(
      isEncoderWorkerRequest({
        type: "init",
        payload: {
          encoder: {
            encoder_id: "proof-encoder",
            encoder_build: "build-1",
            embedding_dim: 4,
            normalized: true,
          },
          modelUrl: "/proof/model.onnx",
          onnxConfigUrl: "/proof/onnx_config.json",
          tokenizerUrl: "/proof/tokenizer.json",
          prefer: "wasm",
        },
      }),
    ).toBe(true);

    expect(
      isEncoderInitResponse({
        type: "encoder_ready",
        state: "ready",
        capabilities: {
          backend: "wasm",
          threaded: false,
          persistentStorage: true,
          encoderId: "proof-encoder",
          encoderBuild: "build-1",
          embeddingDim: 4,
          queryLength: 8,
          doQueryExpansion: false,
          normalized: true,
        },
      }),
    ).toBe(true);

    expect(
      isEncoderInitEvent({
        stage: "config_validated",
        queryLength: 8,
        embeddingDim: 4,
      }),
    ).toBe(true);

    expect(
      isEncodeResponse({
        type: "encoded_query",
        encoded: {
          payload: {
            encoder: {
              encoder_id: "proof-encoder",
              encoder_build: "build-1",
              embedding_dim: 4,
              normalized: true,
            },
            dtype: "f32_le",
            layout: "ragged",
            embeddings: [[0.1, 0.2, 0.3, 0.4]],
          },
          timing: {
            total_ms: 2,
            tokenize_ms: 1,
            inference_ms: 1,
          },
          input_ids: [101, 102],
          attention_mask: [1, 1],
        },
      }),
    ).toBe(true);
  }),
);

it.effect("keeps decodeEncodeResponseSchema aligned with the shared inline payload schema", () =>
  Effect.gen(function*() {
    const response = yield* decodeEncodeResponseSchema({
      type: "encoded_query",
      encoded: {
        payload: {
          encoder: {
            encoder_id: "proof-encoder",
            encoder_build: "build-1",
            embedding_dim: 4,
            normalized: true,
          },
          dtype: "f32_le",
          layout: "padded_query_length",
          embeddings: [[0.1, 0.2, 0.3, 0.4]],
        },
        timing: {
          total_ms: 2,
          tokenize_ms: 1,
          inference_ms: 1,
        },
        input_ids: [101, 102],
        attention_mask: [1, 1],
      },
    });

    expect(response.encoded.payload.layout).toBe("padded_query_length");
    expect(response.encoded.payload.embeddings).toEqual([[0.1, 0.2, 0.3, 0.4]]);
  }),
);

it.effect("rejects encoded query responses with non-finite embedding values", () =>
  Effect.gen(function*() {
    const result = yield* Effect.result(
      decodeEncodeResponseSchema({
        type: "encoded_query",
        encoded: {
          payload: {
            encoder: {
              encoder_id: "proof-encoder",
              encoder_build: "build-1",
              embedding_dim: 4,
              normalized: true,
            },
            dtype: "f32_le",
            layout: "ragged",
            embeddings: [[0.1, Number.NaN, 0.3, 0.4]],
          },
          timing: {
            total_ms: 2,
            tokenize_ms: 1,
            inference_ms: 1,
          },
          input_ids: [101, 102],
          attention_mask: [1, 1],
        },
      }),
    );

    expect(result._tag).toBe("Failure");
  }),
);

it.effect("rejects encoded query responses whose embedding rows do not match the encoder dimension", () =>
  Effect.gen(function*() {
    const result = yield* Effect.result(
      decodeEncodeResponseSchema({
        type: "encoded_query",
        encoded: {
          payload: {
            encoder: {
              encoder_id: "proof-encoder",
              encoder_build: "build-1",
              embedding_dim: 4,
              normalized: true,
            },
            dtype: "f32_le",
            layout: "ragged",
            embeddings: [[0.1, 0.2, 0.3]],
          },
          timing: {
            total_ms: 2,
            tokenize_ms: 1,
            inference_ms: 1,
          },
          input_ids: [101, 102],
          attention_mask: [1, 1],
        },
      }),
    );

    expect(result._tag).toBe("Failure");
  }),
);
