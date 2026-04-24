import { Context, Effect, Layer, Schema } from "effect";

import {
  WorkerRuntimeError,
  type WorkerRuntimeError as WorkerRuntimeErrorType,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import {
  EncoderModelAssetCache,
} from "./encoder-model-asset-cache.js";
import {
  type ModelAssetPackage,
  type ModelAssetPackageKey,
  type ModelAssetStoreKind,
  makeModelAssetEntries,
  makeModelAssetPackageKey,
  modelAssetBytes,
} from "./model-asset-types.js";
import { ModelAssetStore } from "./model-asset-store.js";
import { parseOnnxConfig } from "./onnx-config.js";
import {
  EncoderInitEventSink,
  EncoderRuntimeConfig,
} from "./encoder-runtime-config.js";
import type { EncoderInitEvent, OnnxConfig } from "./types.js";

const decodeJsonString = Schema.decodeUnknownEffect(Schema.UnknownFromJsonString);

type AssetResolutionSource = "memory" | "store" | "network";
type DecodedResolvedAssets = Pick<
  ResolvedEncoderModelAssets,
  "modelBytes" | "tokenizerBytes" | "onnxConfigBytes" | "config"
>;

export interface ResolvedEncoderModelAssets {
  readonly modelBytes: Uint8Array;
  readonly tokenizerBytes: Uint8Array;
  readonly onnxConfigBytes: Uint8Array;
  readonly config: OnnxConfig;
  readonly persistentStorage: boolean;
}

export class EncoderModelAssets
  extends Context.Service<EncoderModelAssets, ResolvedEncoderModelAssets>()(
    "next-plaid-browser/EncoderModelAssets",
  )
{
  static readonly layer = Layer.effect(EncoderModelAssets)(
    makeEncoderModelAssets(),
  );
}

function parseOnnxConfigBytes(
  bytes: Uint8Array,
  url: string,
): Effect.Effect<OnnxConfig, WorkerRuntimeError> {
  return Effect.succeed(new TextDecoder().decode(bytes)).pipe(
    Effect.flatMap((text) =>
      decodeJsonString(text).pipe(
        Effect.mapError((error) =>
          workerRuntimeErrorFromUnknown(
            "encoder_model_assets.parse_onnx_config",
            error,
            `failed to parse onnx config json ${url}`,
          ),
        ),
        Effect.flatMap((value) =>
          Effect.try({
            try: () => parseOnnxConfig(value),
            catch: (error) =>
              workerRuntimeErrorFromUnknown(
                "encoder_model_assets.decode_onnx_config",
                error,
                `failed to decode onnx config ${url}`,
              ),
          }),
        ),
      ),
    ),
  );
}

function decodeResolvedAssets(
  pkg: ModelAssetPackage,
): Effect.Effect<
  DecodedResolvedAssets,
  WorkerRuntimeErrorType
> {
  return Effect.gen(function*() {
    return {
      modelBytes: pkg.modelBytes,
      tokenizerBytes: pkg.tokenizerBytes,
      onnxConfigBytes: pkg.onnxConfigBytes,
      config: yield* parseOnnxConfigBytes(pkg.onnxConfigBytes, pkg.key.onnxConfigUrl),
    };
  });
}

function fetchBytes(
  url: string,
): Effect.Effect<
  {
    readonly bytes: Uint8Array;
    readonly expectedBytes: number | null;
  },
  WorkerRuntimeErrorType
> {
  return Effect.tryPromise({
    try: async () => {
      const response = await fetch(url);
      if (!response.ok) {
        throw workerRuntimeError({
          operation: "encoder_model_assets.fetch_asset",
          message: `failed to fetch asset ${url}: ${response.status} ${response.statusText}`,
          details: { url },
        });
      }

      const expectedBytesHeader = response.headers.get("content-length");
      const expectedBytes =
        expectedBytesHeader === null ? null : Number(expectedBytesHeader);

      return {
        bytes: new Uint8Array(await response.arrayBuffer()),
        expectedBytes:
          expectedBytes !== null && Number.isFinite(expectedBytes)
            ? expectedBytes
            : null,
      };
    },
    catch: (error) =>
      error instanceof WorkerRuntimeError
        ? error
        : workerRuntimeErrorFromUnknown(
            "encoder_model_assets.fetch_asset",
            error,
            `failed to fetch asset ${url}`,
          ),
  });
}

function emitMemoryHit(
  key: ModelAssetPackageKey,
  pkg: ModelAssetPackage,
  emit: (event: EncoderInitEvent) => Effect.Effect<void>,
): Effect.Effect<void> {
  return Effect.forEach(makeModelAssetEntries(key), (entry) =>
    emit({
      stage: "asset_memory_hit",
      url: entry.url,
      bytesReceived: modelAssetBytes(pkg, entry.kind).byteLength,
    }),
  ).pipe(Effect.asVoid);
}

function emitStoreHit(
  key: ModelAssetPackageKey,
  pkg: ModelAssetPackage,
  storeKind: ModelAssetStoreKind,
  emit: (event: EncoderInitEvent) => Effect.Effect<void>,
): Effect.Effect<void> {
  return Effect.forEach(makeModelAssetEntries(key), (entry) =>
    emit({
      stage: "asset_store_hit",
      url: entry.url,
      storeKind,
      bytesReceived: modelAssetBytes(pkg, entry.kind).byteLength,
    }),
  ).pipe(Effect.asVoid);
}

function emitStoreMiss(
  key: ModelAssetPackageKey,
  storeKind: ModelAssetStoreKind,
  emit: (event: EncoderInitEvent) => Effect.Effect<void>,
): Effect.Effect<void> {
  return Effect.forEach(makeModelAssetEntries(key), (entry) =>
    emit({
      stage: "asset_store_miss",
      url: entry.url,
      storeKind,
    }),
  ).pipe(Effect.asVoid);
}

function emitStoreWrite(
  stage: "asset_store_write_start" | "asset_store_write_complete",
  key: ModelAssetPackageKey,
  pkg: ModelAssetPackage,
  storeKind: Exclude<ModelAssetStoreKind, "transient">,
  emit: (event: EncoderInitEvent) => Effect.Effect<void>,
): Effect.Effect<void> {
  return Effect.forEach(makeModelAssetEntries(key), (entry) =>
    emit({
      stage,
      url: entry.url,
      storeKind,
      bytesReceived: modelAssetBytes(pkg, entry.kind).byteLength,
    }),
  ).pipe(Effect.asVoid);
}

function fetchNetworkPackage(
  key: ModelAssetPackageKey,
  emit: (event: EncoderInitEvent) => Effect.Effect<void>,
): Effect.Effect<ModelAssetPackage, WorkerRuntimeErrorType> {
  return Effect.gen(function*() {
    const bytesByKind = new Map<
      "model" | "tokenizer" | "onnxConfig",
      Uint8Array
    >();

    for (const entry of makeModelAssetEntries(key)) {
      yield* emit({
        stage: "asset_fetch_start",
        url: entry.url,
        expectedBytes: null,
      });
      const response = yield* fetchBytes(entry.url);
      yield* emit({
        stage: "asset_fetch_complete",
        url: entry.url,
        bytesReceived: response.bytes.byteLength,
      });
      bytesByKind.set(entry.kind, response.bytes);
    }

    const modelBytes = bytesByKind.get("model");
    const tokenizerBytes = bytesByKind.get("tokenizer");
    const onnxConfigBytes = bytesByKind.get("onnxConfig");

    if (
      modelBytes === undefined ||
      tokenizerBytes === undefined ||
      onnxConfigBytes === undefined
    ) {
      return yield* workerRuntimeError({
        operation: "encoder_model_assets.fetch_package",
        message: `failed to resolve all fetched assets for package ${key.packageId}`,
      });
    }

    return {
      key,
      modelBytes,
      tokenizerBytes,
      onnxConfigBytes,
    };
  });
}

function makeEncoderModelAssets(): Effect.Effect<
  ResolvedEncoderModelAssets,
  WorkerRuntimeErrorType,
  EncoderRuntimeConfig | EncoderInitEventSink | EncoderModelAssetCache | ModelAssetStore
> {
  return Effect.gen(function*() {
    const { input } = yield* EncoderRuntimeConfig;
    const eventSink = yield* EncoderInitEventSink;
    const memoryCache = yield* EncoderModelAssetCache;
    const assetStore = yield* ModelAssetStore;
    const key = yield* makeModelAssetPackageKey(input);
    const persistentStorage = yield* assetStore.persistentStorage();

    const resolvePackage = (
      options?: {
        readonly skipStore?: boolean;
      },
    ): Effect.Effect<
      {
        readonly pkg: ModelAssetPackage;
        readonly source: AssetResolutionSource;
        readonly decoded: DecodedResolvedAssets | null;
      },
      WorkerRuntimeError
    > =>
      Effect.gen(function*() {
        const cachedPackage = yield* memoryCache.get(key.packageId);
        if (cachedPackage !== null) {
          yield* emitMemoryHit(key, cachedPackage, eventSink.emit);
          return {
            pkg: cachedPackage,
            source: "memory",
            decoded: null,
          } as const;
        }

        const storeKind = yield* assetStore.kind();
        if (options?.skipStore !== true) {
          const storedPackage = yield* assetStore.loadPackage(key);
          if (storedPackage !== null) {
            yield* emitStoreHit(key, storedPackage, storeKind, eventSink.emit);
            yield* memoryCache.put(storedPackage);
            return {
              pkg: storedPackage,
              source: "store",
              decoded: null,
            } as const;
          }
        }

        yield* emitStoreMiss(key, storeKind, eventSink.emit);
        const fetchedPackage = yield* fetchNetworkPackage(key, eventSink.emit);

        // Parse the structured assets before persisting so we never commit a bad package.
        const decoded = yield* decodeResolvedAssets(fetchedPackage);

        if (storeKind !== "transient") {
          yield* emitStoreWrite(
            "asset_store_write_start",
            key,
            fetchedPackage,
            storeKind,
            eventSink.emit,
          );
        }
        yield* assetStore.storePackage(fetchedPackage);
        if (storeKind !== "transient") {
          yield* emitStoreWrite(
            "asset_store_write_complete",
            key,
            fetchedPackage,
            storeKind,
            eventSink.emit,
          );
        }

        yield* memoryCache.put(fetchedPackage);
        return {
          pkg: fetchedPackage,
          source: "network",
          decoded,
        } as const;
      });

    const initialResolution = yield* resolvePackage();

    const resolved = yield* (initialResolution.decoded === null
      ? decodeResolvedAssets(initialResolution.pkg)
      : Effect.succeed(initialResolution.decoded)).pipe(
      Effect.catchTag("WorkerRuntimeError", (error) =>
        initialResolution.source === "store"
          ? Effect.gen(function*() {
              yield* memoryCache.remove(key.packageId);
              yield* assetStore.removePackage(key.packageId).pipe(
                Effect.orElseSucceed(() => void 0),
              );
              const refetched = yield* resolvePackage({ skipStore: true });
              if (refetched.source === "memory") {
                return yield* error;
              }
              return yield* (refetched.decoded === null
                ? decodeResolvedAssets(refetched.pkg)
                : Effect.succeed(refetched.decoded));
            })
          : Effect.gen(function*() {
              return yield* error;
            }),
      ),
    );

    return EncoderModelAssets.of({
      ...resolved,
      persistentStorage,
    });
  });
}
