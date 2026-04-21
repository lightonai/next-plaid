import {
  Cache as EffectCache,
  Context,
  Duration,
  Effect,
  Layer,
} from "effect";

import {
  WorkerRuntimeError,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";

const MODEL_CACHE_NAME = "next-plaid-browser-model-worker-v1";
const textDecoder = new TextDecoder();
const ASSET_CACHE_CAPACITY = 16;

export interface EncoderModelAssetCacheApi {
  readonly readCachedBytes: (
    url: string,
  ) => Effect.Effect<Uint8Array | null, WorkerRuntimeError>;
  readonly fetchAndCacheBytes: (
    url: string,
  ) => Effect.Effect<Uint8Array, WorkerRuntimeError>;
  readonly loadBytes: (url: string) => Effect.Effect<Uint8Array, WorkerRuntimeError>;
  readonly loadText: (url: string) => Effect.Effect<string, WorkerRuntimeError>;
  readonly persistentStorage: () => Effect.Effect<boolean, WorkerRuntimeError>;
}

export class EncoderModelAssetCache
  extends Context.Service<EncoderModelAssetCache, EncoderModelAssetCacheApi>()(
    "next-plaid-browser/EncoderModelAssetCache",
  )
{
  static readonly layer = Layer.effect(EncoderModelAssetCache)(
    makeEncoderModelAssetCache(),
  );
}

function fetchOk(url: string): Effect.Effect<Response, WorkerRuntimeError> {
  return Effect.tryPromise({
    try: async () => {
      const response = await fetch(url);
      if (!response.ok) {
        throw workerRuntimeError({
          operation: "encoder_model_asset_cache.fetch_asset",
          message: `failed to fetch asset ${url}: ${response.status} ${response.statusText}`,
          details: { url },
        });
      }
      return response;
    },
    catch: (error) =>
      error instanceof WorkerRuntimeError
        ? error
        : workerRuntimeErrorFromUnknown(
            "encoder_model_asset_cache.fetch_asset",
            error,
            `failed to fetch asset ${url}`,
          ),
  });
}

function openPersistentAssetCache(): Effect.Effect<
  globalThis.Cache | null,
  WorkerRuntimeError
> {
  if (typeof caches === "undefined") {
    return Effect.succeed(null);
  }

  return Effect.tryPromise({
    try: () => caches.open(MODEL_CACHE_NAME),
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "encoder_model_asset_cache.open_cache",
        error,
        "failed to open model cache",
      ),
  });
}

function queryPersistentStorage(): Effect.Effect<boolean, WorkerRuntimeError> {
  if (
    typeof navigator === "undefined" ||
    !("storage" in navigator) ||
    typeof navigator.storage?.persisted !== "function"
  ) {
    return Effect.succeed(false);
  }

  return Effect.tryPromise({
    try: () => navigator.storage.persisted(),
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "encoder_model_asset_cache.persistent_storage",
        error,
        "failed to query persistent storage state",
      ),
  });
}

function readBytes(
  response: Response,
  url: string,
): Effect.Effect<Uint8Array, WorkerRuntimeError> {
  return Effect.tryPromise({
    try: async () => new Uint8Array(await response.arrayBuffer()),
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "encoder_model_asset_cache.read_asset_bytes",
        error,
        `failed to read bytes for ${url}`,
      ),
  });
}

function readCachedResponse(
  browserCache: globalThis.Cache | null,
  url: string,
): Effect.Effect<Response | null, WorkerRuntimeError> {
  if (browserCache === null) {
    return Effect.succeed(null);
  }

  return Effect.tryPromise({
    try: async () => (await browserCache.match(url)) ?? null,
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "encoder_model_asset_cache.read_cache",
        error,
        `failed to read cached model asset ${url}`,
      ),
  });
}

function persistResponse(
  browserCache: globalThis.Cache | null,
  url: string,
  response: Response,
): Effect.Effect<void, WorkerRuntimeError> {
  if (browserCache === null) {
    return Effect.void;
  }

  return Effect.tryPromise({
    try: () => browserCache.put(url, response.clone()),
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "encoder_model_asset_cache.write_cache",
        error,
        `failed to cache model asset ${url}`,
      ),
  });
}

function makeEncoderModelAssetCache(): Effect.Effect<
  EncoderModelAssetCacheApi,
  never
> {
  return Effect.gen(function*() {
    const openPersistentCache = yield* openPersistentAssetCache().pipe(
      Effect.cachedWithTTL(Duration.infinity),
    );
    const readPersistentStorage = yield* queryPersistentStorage().pipe(
      Effect.cachedWithTTL(Duration.infinity),
    );
    const readCachedBytes_ = Effect.fn(
      "EncoderModelAssetCache.readCachedBytes",
    )(function*(url: string) {
      const browserCache = yield* openPersistentCache;
      const cachedResponse = yield* readCachedResponse(browserCache, url);
      if (cachedResponse === null) {
        return null;
      }
      return yield* readBytes(cachedResponse, url);
    });
    const fetchAndCacheBytes_ = Effect.fn(
      "EncoderModelAssetCache.fetchAndCacheBytes",
    )(function*(url: string) {
      const browserCache = yield* openPersistentCache;
      const response = yield* fetchOk(url);
      yield* persistResponse(browserCache, url, response);
      return yield* readBytes(response, url);
    });
    const bytesCache = yield* EffectCache.make<string, Uint8Array, WorkerRuntimeError>({
      capacity: ASSET_CACHE_CAPACITY,
      lookup: (url) =>
        Effect.gen(function*() {
          const cachedBytes = yield* readCachedBytes_(url);
          if (cachedBytes !== null) {
            return cachedBytes;
          }

          return yield* fetchAndCacheBytes_(url);
        }),
    });

    const loadBytes = Effect.fn("EncoderModelAssetCache.loadBytes")(
      function*(url: string) {
        return yield* EffectCache.get(bytesCache, url);
      },
    );
    const loadText = Effect.fn("EncoderModelAssetCache.loadText")(
      function*(url: string) {
        const bytes = yield* loadBytes(url);
        return textDecoder.decode(bytes);
      },
    );
    const persistentStorage = Effect.fn(
      "EncoderModelAssetCache.persistentStorage",
    )(function*() {
      return yield* readPersistentStorage;
    });

    return EncoderModelAssetCache.of({
      readCachedBytes: readCachedBytes_,
      fetchAndCacheBytes: fetchAndCacheBytes_,
      loadBytes,
      loadText,
      persistentStorage,
    });
  });
}
