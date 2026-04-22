import { Context, Effect, Layer, Ref } from "effect";

import type { ModelAssetPackage } from "./model-asset-types.js";

export interface EncoderModelAssetCacheApi {
  readonly get: (
    packageId: string,
  ) => Effect.Effect<ModelAssetPackage | null>;
  readonly put: (
    pkg: ModelAssetPackage,
  ) => Effect.Effect<void>;
  readonly remove: (
    packageId: string,
  ) => Effect.Effect<void>;
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

function makeEncoderModelAssetCache(): Effect.Effect<
  EncoderModelAssetCacheApi,
  never
> {
  return Effect.gen(function*() {
    const cacheRef = yield* Ref.make(new Map<string, ModelAssetPackage>());

    const get = Effect.fn("EncoderModelAssetCache.get")(function*(packageId: string) {
      const cache = yield* Ref.get(cacheRef);
      return cache.get(packageId) ?? null;
    });

    const put = Effect.fn("EncoderModelAssetCache.put")(function*(pkg: ModelAssetPackage) {
      yield* Ref.update(cacheRef, (cache) => {
        const next = new Map(cache);
        next.set(pkg.key.packageId, pkg);
        return next;
      });
    });

    const remove = Effect.fn("EncoderModelAssetCache.remove")(function*(packageId: string) {
      yield* Ref.update(cacheRef, (cache) => {
        if (!cache.has(packageId)) {
          return cache;
        }
        const next = new Map(cache);
        next.delete(packageId);
        return next;
      });
    });

    return EncoderModelAssetCache.of({
      get,
      put,
      remove,
    });
  });
}
