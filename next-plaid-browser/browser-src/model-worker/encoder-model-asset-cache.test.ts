import { expect, it } from "@effect/vitest";
import { Context, Effect, Layer, Scope } from "effect";
import * as Exit from "effect/Exit";

import { EncoderModelAssetCache } from "./encoder-model-asset-cache.js";
import type { ModelAssetPackage } from "./model-asset-types.js";

function modelAssetPackage(packageId = "pkg-1"): ModelAssetPackage {
  return {
    key: {
      packageId,
      encoderId: "proof-encoder",
      encoderBuild: "build-1",
      embeddingDim: 4,
      normalized: true,
      modelUrl: "https://example.test/model.onnx",
      tokenizerUrl: "https://example.test/tokenizer.json",
      onnxConfigUrl: "https://example.test/onnx_config.json",
    },
    modelBytes: new Uint8Array([1, 2, 3]),
    tokenizerBytes: new Uint8Array([4, 5, 6]),
    onnxConfigBytes: new Uint8Array([7, 8, 9]),
  };
}

it.effect("memoizes resolved model packages within one service lifetime", () =>
  Effect.gen(function*() {
    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

    const context = yield* Layer.buildWithScope(
      EncoderModelAssetCache.layer,
      scope,
    );
    const assetCache = Context.get(context, EncoderModelAssetCache);
    const pkg = modelAssetPackage();

    yield* assetCache.put(pkg);
    const first = yield* assetCache.get(pkg.key.packageId);
    const second = yield* assetCache.get(pkg.key.packageId);

    expect(first).toEqual(pkg);
    expect(second).toEqual(pkg);
  }),
);

it.effect("removes cached model packages by package id", () =>
  Effect.gen(function*() {
    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

    const context = yield* Layer.buildWithScope(
      EncoderModelAssetCache.layer,
      scope,
    );
    const assetCache = Context.get(context, EncoderModelAssetCache);
    const pkg = modelAssetPackage();

    yield* assetCache.put(pkg);
    yield* assetCache.remove(pkg.key.packageId);
    const value = yield* assetCache.get(pkg.key.packageId);

    expect(value).toBeNull();
  }),
);

it.effect("evicts older model packages when capacity is exceeded", () =>
  Effect.gen(function*() {
    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

    const context = yield* Layer.buildWithScope(
      EncoderModelAssetCache.layerWithCapacity(2),
      scope,
    );
    const assetCache = Context.get(context, EncoderModelAssetCache);
    const first = modelAssetPackage("pkg-1");
    const second = modelAssetPackage("pkg-2");
    const third = modelAssetPackage("pkg-3");

    yield* assetCache.put(first);
    yield* assetCache.put(second);
    yield* assetCache.put(third);

    expect(yield* assetCache.get(first.key.packageId)).toBeNull();
    expect(yield* assetCache.get(second.key.packageId)).toEqual(second);
    expect(yield* assetCache.get(third.key.packageId)).toEqual(third);
  }),
);
