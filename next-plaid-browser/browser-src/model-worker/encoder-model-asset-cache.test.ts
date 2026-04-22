import { expect, it } from "@effect/vitest";
import { Context, Effect, Layer, Scope } from "effect";
import * as Exit from "effect/Exit";

import { EncoderModelAssetCache } from "./encoder-model-asset-cache.js";
import type { EncoderInitEvent } from "./types.js";

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

it.effect("loads assets through one telemetry boundary and reports cache hits after persistence reuse", () =>
  Effect.gen(function*() {
    const originalFetch = globalThis.fetch;
    const hadCaches = Reflect.has(globalThis, "caches");
    const originalCaches = hadCaches ? Reflect.get(globalThis, "caches") : undefined;
    let fetchCount = 0;
    const cachedResponses = new Map<string, Response>();

    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        globalThis.fetch = originalFetch;
        if (hadCaches) {
          Reflect.set(globalThis, "caches", originalCaches);
        } else {
          Reflect.deleteProperty(globalThis, "caches");
        }
      }),
    );

    Reflect.set(globalThis, "caches", {
      open: async () => ({
        match: async (url: string) => cachedResponses.get(url)?.clone() ?? null,
        put: async (url: string, response: Response) => {
          cachedResponses.set(url, response.clone());
        },
      }),
    });

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
    const events: EncoderInitEvent[] = [];
    const url = "https://example.test/model.onnx";

    const first = yield* assetCache.loadBytesWithTelemetry(url, {
      emit: (event) =>
        Effect.sync(() => {
          events.push(event);
        }),
    });
    const second = yield* assetCache.loadBytesWithTelemetry(url, {
      emit: (event) =>
        Effect.sync(() => {
          events.push(event);
        }),
    });

    expect(new TextDecoder().decode(first)).toBe("asset-payload");
    expect(new TextDecoder().decode(second)).toBe("asset-payload");
    expect(fetchCount).toBe(1);
    expect(events).toEqual([
      { stage: "asset_cache_miss", url },
      { stage: "asset_fetch_start", url, expectedBytes: null },
      {
        stage: "asset_fetch_complete",
        url,
        bytesReceived: first.byteLength,
      },
      {
        stage: "asset_cache_hit",
        url,
        bytesReceived: second.byteLength,
      },
    ]);
  }),
);
