import { expect, it } from "@effect/vitest";
import { Effect } from "effect";

import { makeModelAssetPackageKey } from "./model-asset-types.js";

function modelAssetIdentityInput(overrides?: {
  readonly encoderBuild?: string;
  readonly modelUrl?: string;
  readonly tokenizerUrl?: string;
  readonly onnxConfigUrl?: string;
}) {
  return {
    encoder: {
      encoder_id: "proof-encoder",
      encoder_build: overrides?.encoderBuild ?? "proof-build-1",
      embedding_dim: 4,
      normalized: true,
    },
    modelUrl: overrides?.modelUrl ?? "https://example.test/model.onnx",
    tokenizerUrl:
      overrides?.tokenizerUrl ?? "https://example.test/tokenizer.json",
    onnxConfigUrl:
      overrides?.onnxConfigUrl ?? "https://example.test/onnx_config.json",
  };
}

it.effect("changes the package id when the encoder build changes", () =>
  Effect.gen(function* () {
    const first = yield* makeModelAssetPackageKey(modelAssetIdentityInput());
    const second = yield* makeModelAssetPackageKey(
      modelAssetIdentityInput({
        encoderBuild: "proof-build-2",
      }),
    );

    expect(first.packageId).not.toBe(second.packageId);
  }),
);

it.effect("changes the package id when any asset URL changes", () =>
  Effect.gen(function* () {
    const first = yield* makeModelAssetPackageKey(modelAssetIdentityInput());
    const second = yield* makeModelAssetPackageKey(
      modelAssetIdentityInput({
        modelUrl: "https://example.test/model-v2.onnx",
      }),
    );

    expect(first.packageId).not.toBe(second.packageId);
  }),
);
