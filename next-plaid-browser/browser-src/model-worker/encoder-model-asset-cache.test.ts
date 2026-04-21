import { expect, it } from "@effect/vitest";
import { Context, Effect, Layer, Scope } from "effect";
import * as Exit from "effect/Exit";

import { EncoderModelAssetCache } from "./encoder-model-asset-cache.js";

it.effect("memoizes repeated asset reads within one service lifetime", () =>
  Effect.gen(function*() {
    const originalFetch = globalThis.fetch;
    let fetchCount = 0;
    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        globalThis.fetch = originalFetch;
      }),
    );

    globalThis.fetch = (((_input: string | URL | Request, _init?: RequestInit) => {
      fetchCount += 1;
      return Promise.resolve(
        new Response("asset-payload", {
          status: 200,
          headers: {
            "content-type": "application/octet-stream",
          },
        }),
      );
    }) as typeof fetch);

    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

    const context = yield* Layer.buildWithScope(
      EncoderModelAssetCache.layer,
      scope,
    );
    const assetCache = Context.get(context, EncoderModelAssetCache);

    const first = yield* assetCache.loadText("https://example.test/model.onnx");
    const second = yield* assetCache.loadText("https://example.test/model.onnx");

    expect(first).toBe("asset-payload");
    expect(second).toBe("asset-payload");
    expect(fetchCount).toBe(1);
  }),
);
