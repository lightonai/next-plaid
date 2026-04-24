import { expect, it } from "@effect/vitest";
import { IDBFactory, IDBKeyRange } from "fake-indexeddb";
import { Context, Effect, Layer, Ref, Scope } from "effect";
import * as Exit from "effect/Exit";

import { EncoderModelAssetCache } from "./encoder-model-asset-cache.js";
import {
  EncoderModelAssets,
  type ResolvedEncoderModelAssets,
} from "./encoder-model-assets.js";
import { makeModelAssetPackageKey } from "./model-asset-types.js";
import {
  EncoderInitEventSink,
  EncoderRuntimeConfig,
} from "./encoder-runtime-config.js";
import { ModelAssetStore } from "./model-asset-store.js";
import type {
  EncoderCreateInput,
  EncoderInitEvent,
} from "./types.js";

function encoderInput(): EncoderCreateInput {
  return {
    encoder: {
      encoder_id: "proof-encoder",
      encoder_build: "proof-build-1",
      embedding_dim: 4,
      normalized: true,
    },
    modelUrl: "https://example.test/model.onnx",
    tokenizerUrl: "https://example.test/tokenizer.json",
    onnxConfigUrl: "https://example.test/onnx_config.json",
    prefer: "wasm",
  };
}

function tokenizerBytes(): Uint8Array {
  return new TextEncoder().encode(
    JSON.stringify({
      kind: "whitespace_vocab",
      version: 1,
      vocab: {},
      unknown_token_id: 0,
    }),
  );
}

function onnxConfigBytes(): Uint8Array {
  return new TextEncoder().encode(
    JSON.stringify({
      query_prefix: "[Q] ",
      document_prefix: "[D] ",
      query_length: 8,
      document_length: 16,
      do_query_expansion: false,
      embedding_dim: 4,
      uses_token_type_ids: false,
      mask_token_id: 103,
      pad_token_id: 0,
      skiplist_words: [],
      do_lower_case: false,
    }),
  );
}

function patchFetch(
  responses: Readonly<Record<string, Uint8Array>>,
): {
  readonly calls: Array<string>;
  readonly restore: () => void;
} {
  const originalFetch = globalThis.fetch;
  const calls: Array<string> = [];

  globalThis.fetch = (async (input: string | URL | Request) => {
    const url = typeof input === "string"
      ? input
      : input instanceof URL
        ? input.toString()
        : input.url;
    calls.push(url);
    const bytes = responses[url];
    if (bytes === undefined) {
      return new Response("missing test fixture", { status: 404 });
    }
    const body = Uint8Array.from(bytes);
    return new Response(
      body.buffer as ArrayBuffer,
      {
        status: 200,
        headers: {
          "content-length": String(bytes.byteLength),
        },
      },
    );
  }) as typeof fetch;

  return {
    calls,
    restore: () => {
      globalThis.fetch = originalFetch;
    },
  };
}

function patchIndexedDbGlobals(): {
  readonly restore: () => void;
} {
  const originalIndexedDb = (globalThis as Record<string, unknown>).indexedDB;
  const originalIDBKeyRange = (globalThis as Record<string, unknown>).IDBKeyRange;

  Object.defineProperty(globalThis, "indexedDB", {
    configurable: true,
    value: new IDBFactory(),
  });
  Object.defineProperty(globalThis, "IDBKeyRange", {
    configurable: true,
    value: IDBKeyRange,
  });

  return {
    restore: () => {
      Object.defineProperty(globalThis, "indexedDB", {
        configurable: true,
        value: originalIndexedDb,
      });
      Object.defineProperty(globalThis, "IDBKeyRange", {
        configurable: true,
        value: originalIDBKeyRange,
      });
    },
  };
}

function buildResolvedAssets(
  scope: Scope.Scope,
  dependencies: Context.Context<
    | EncoderRuntimeConfig
    | EncoderInitEventSink
    | EncoderModelAssetCache
    | ModelAssetStore
  >,
): Effect.Effect<ResolvedEncoderModelAssets, never> {
  return Layer.buildWithScope(
    Layer.fresh(EncoderModelAssets.layer).pipe(
      Layer.provide(Layer.succeedContext(dependencies)),
    ),
    scope,
  ).pipe(
    Effect.map((context) => Context.get(context, EncoderModelAssets)),
  ) as Effect.Effect<ResolvedEncoderModelAssets, never>;
}

it.effect("reuses the worker memory cache before consulting the store again", () =>
  Effect.gen(function* () {
    const fetchStub = patchFetch({
      [encoderInput().modelUrl]: new Uint8Array([1, 2, 3, 4]),
      [encoderInput().tokenizerUrl]: tokenizerBytes(),
      [encoderInput().onnxConfigUrl]: onnxConfigBytes(),
    });
    yield* Effect.addFinalizer(() => Effect.sync(fetchStub.restore));

    const eventsRef = yield* Ref.make<ReadonlyArray<EncoderInitEvent>>([]);
    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

    const dependencyContext = yield* Layer.buildWithScope(
      Layer.mergeAll(
        EncoderRuntimeConfig.layer(encoderInput()),
        EncoderInitEventSink.layer((event) =>
          Ref.update(eventsRef, (events) => [...events, event]),
        ),
        EncoderModelAssetCache.layer,
        ModelAssetStore.layerTransient,
      ),
      scope,
    );

    const first = yield* buildResolvedAssets(scope, dependencyContext);
    yield* Ref.set(eventsRef, []);
    const second = yield* buildResolvedAssets(scope, dependencyContext);
    const secondPassEvents = yield* Ref.get(eventsRef);

    expect(fetchStub.calls).toEqual([
      encoderInput().modelUrl,
      encoderInput().tokenizerUrl,
      encoderInput().onnxConfigUrl,
    ]);
    expect(secondPassEvents).toEqual([
      {
        stage: "asset_memory_hit",
        url: encoderInput().modelUrl,
        bytesReceived: 4,
      },
      {
        stage: "asset_memory_hit",
        url: encoderInput().tokenizerUrl,
        bytesReceived: tokenizerBytes().byteLength,
      },
      {
        stage: "asset_memory_hit",
        url: encoderInput().onnxConfigUrl,
        bytesReceived: onnxConfigBytes().byteLength,
      },
    ]);
    expect(second.modelBytes).toEqual(first.modelBytes);
    expect(second.config).toEqual(first.config);
  }),
);

it.effect("removes a corrupt stored package and refetches it from the network", () =>
  Effect.gen(function* () {
    const fetchStub = patchFetch({
      [encoderInput().modelUrl]: new Uint8Array([1, 2, 3, 4]),
      [encoderInput().tokenizerUrl]: tokenizerBytes(),
      [encoderInput().onnxConfigUrl]: onnxConfigBytes(),
    });
    yield* Effect.addFinalizer(() => Effect.sync(fetchStub.restore));
    const corruptOnnxConfigBytes = new TextEncoder().encode("{not valid json");

    const eventsRef = yield* Ref.make<ReadonlyArray<EncoderInitEvent>>([]);
    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

    const dependencyContext = yield* Layer.buildWithScope(
      Layer.mergeAll(
        EncoderRuntimeConfig.layer(encoderInput()),
        EncoderInitEventSink.layer((event) =>
          Ref.update(eventsRef, (events) => [...events, event]),
        ),
        EncoderModelAssetCache.layer,
        ModelAssetStore.layerTransient,
      ),
      scope,
    );

    const assetStore = Context.get(dependencyContext, ModelAssetStore);
    const packageKey = yield* makeModelAssetPackageKey(encoderInput());
    yield* assetStore.storePackage({
      key: packageKey,
      modelBytes: new Uint8Array([1, 2, 3, 4]),
      tokenizerBytes: tokenizerBytes(),
      onnxConfigBytes: corruptOnnxConfigBytes,
    });

    const first = yield* buildResolvedAssets(scope, dependencyContext);
    const firstEvents = yield* Ref.get(eventsRef);
    yield* Ref.set(eventsRef, []);
    const second = yield* buildResolvedAssets(scope, dependencyContext);
    const secondEvents = yield* Ref.get(eventsRef);

    expect(fetchStub.calls).toEqual([
      encoderInput().modelUrl,
      encoderInput().tokenizerUrl,
      encoderInput().onnxConfigUrl,
    ]);
    expect(first.persistentStorage).toBe(false);
    expect(first.config).toEqual(second.config);
    expect(firstEvents).toEqual([
      {
        stage: "asset_store_hit",
        url: encoderInput().modelUrl,
        storeKind: "transient",
        bytesReceived: 4,
      },
      {
        stage: "asset_store_hit",
        url: encoderInput().tokenizerUrl,
        storeKind: "transient",
        bytesReceived: tokenizerBytes().byteLength,
      },
      {
        stage: "asset_store_hit",
        url: encoderInput().onnxConfigUrl,
        storeKind: "transient",
        bytesReceived: corruptOnnxConfigBytes.byteLength,
      },
      {
        stage: "asset_store_miss",
        url: encoderInput().modelUrl,
        storeKind: "transient",
      },
      {
        stage: "asset_store_miss",
        url: encoderInput().tokenizerUrl,
        storeKind: "transient",
      },
      {
        stage: "asset_store_miss",
        url: encoderInput().onnxConfigUrl,
        storeKind: "transient",
      },
      {
        stage: "asset_fetch_start",
        url: encoderInput().modelUrl,
        expectedBytes: null,
      },
      {
        stage: "asset_fetch_complete",
        url: encoderInput().modelUrl,
        bytesReceived: 4,
      },
      {
        stage: "asset_fetch_start",
        url: encoderInput().tokenizerUrl,
        expectedBytes: null,
      },
      {
        stage: "asset_fetch_complete",
        url: encoderInput().tokenizerUrl,
        bytesReceived: tokenizerBytes().byteLength,
      },
      {
        stage: "asset_fetch_start",
        url: encoderInput().onnxConfigUrl,
        expectedBytes: null,
      },
      {
        stage: "asset_fetch_complete",
        url: encoderInput().onnxConfigUrl,
        bytesReceived: onnxConfigBytes().byteLength,
      },
    ]);
    expect(secondEvents).toEqual([
      {
        stage: "asset_memory_hit",
        url: encoderInput().modelUrl,
        bytesReceived: 4,
      },
      {
        stage: "asset_memory_hit",
        url: encoderInput().tokenizerUrl,
        bytesReceived: tokenizerBytes().byteLength,
      },
      {
        stage: "asset_memory_hit",
        url: encoderInput().onnxConfigUrl,
        bytesReceived: onnxConfigBytes().byteLength,
      },
    ]);
  }),
);

it.effect("emits the real store-miss fetch and write order for durable package installs", () =>
  Effect.gen(function* () {
    const globals = patchIndexedDbGlobals();
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const fetchStub = patchFetch({
      [encoderInput().modelUrl]: new Uint8Array([1, 2, 3, 4]),
      [encoderInput().tokenizerUrl]: tokenizerBytes(),
      [encoderInput().onnxConfigUrl]: onnxConfigBytes(),
    });
    yield* Effect.addFinalizer(() => Effect.sync(fetchStub.restore));

    const eventsRef = yield* Ref.make<ReadonlyArray<EncoderInitEvent>>([]);
    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

    const dependencyContext = yield* Layer.buildWithScope(
      Layer.mergeAll(
        EncoderRuntimeConfig.layer(encoderInput()),
        EncoderInitEventSink.layer((event) =>
          Ref.update(eventsRef, (events) => [...events, event]),
        ),
        EncoderModelAssetCache.layer,
        ModelAssetStore.layerIndexedDb,
      ),
      scope,
    );

    const resolved = yield* buildResolvedAssets(scope, dependencyContext);
    const events = yield* Ref.get(eventsRef);

    expect(resolved.persistentStorage).toBe(true);
    expect(fetchStub.calls).toEqual([
      encoderInput().modelUrl,
      encoderInput().tokenizerUrl,
      encoderInput().onnxConfigUrl,
    ]);
    expect(events).toEqual([
      {
        stage: "asset_store_miss",
        url: encoderInput().modelUrl,
        storeKind: "indexeddb",
      },
      {
        stage: "asset_store_miss",
        url: encoderInput().tokenizerUrl,
        storeKind: "indexeddb",
      },
      {
        stage: "asset_store_miss",
        url: encoderInput().onnxConfigUrl,
        storeKind: "indexeddb",
      },
      {
        stage: "asset_fetch_start",
        url: encoderInput().modelUrl,
        expectedBytes: null,
      },
      {
        stage: "asset_fetch_complete",
        url: encoderInput().modelUrl,
        bytesReceived: 4,
      },
      {
        stage: "asset_fetch_start",
        url: encoderInput().tokenizerUrl,
        expectedBytes: null,
      },
      {
        stage: "asset_fetch_complete",
        url: encoderInput().tokenizerUrl,
        bytesReceived: tokenizerBytes().byteLength,
      },
      {
        stage: "asset_fetch_start",
        url: encoderInput().onnxConfigUrl,
        expectedBytes: null,
      },
      {
        stage: "asset_fetch_complete",
        url: encoderInput().onnxConfigUrl,
        bytesReceived: onnxConfigBytes().byteLength,
      },
      {
        stage: "asset_store_write_start",
        url: encoderInput().modelUrl,
        storeKind: "indexeddb",
        bytesReceived: 4,
      },
      {
        stage: "asset_store_write_start",
        url: encoderInput().tokenizerUrl,
        storeKind: "indexeddb",
        bytesReceived: tokenizerBytes().byteLength,
      },
      {
        stage: "asset_store_write_start",
        url: encoderInput().onnxConfigUrl,
        storeKind: "indexeddb",
        bytesReceived: onnxConfigBytes().byteLength,
      },
      {
        stage: "asset_store_write_complete",
        url: encoderInput().modelUrl,
        storeKind: "indexeddb",
        bytesReceived: 4,
      },
      {
        stage: "asset_store_write_complete",
        url: encoderInput().tokenizerUrl,
        storeKind: "indexeddb",
        bytesReceived: tokenizerBytes().byteLength,
      },
      {
        stage: "asset_store_write_complete",
        url: encoderInput().onnxConfigUrl,
        storeKind: "indexeddb",
        bytesReceived: onnxConfigBytes().byteLength,
      },
    ]);
  }),
);
