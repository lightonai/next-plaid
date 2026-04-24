import { Cache as EffectCache, Context, Duration, Effect, Layer, Option } from "effect";

import type { ModelAssetPackage } from "./model-asset-types.js";

const DEFAULT_MODEL_ASSET_CACHE_CAPACITY = 3;

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
  static layerWithCapacity(capacity: number) {
    return Layer.effect(EncoderModelAssetCache)(
      makeEncoderModelAssetCache({ capacity }),
    );
  }

  static readonly layer = EncoderModelAssetCache.layerWithCapacity(
    DEFAULT_MODEL_ASSET_CACHE_CAPACITY,
  );
}

function makeEncoderModelAssetCache(options: {
  readonly capacity: number;
}): Effect.Effect<
  EncoderModelAssetCacheApi,
  never
> {
  return Effect.gen(function*() {
    const cache = yield* EffectCache.make<string, ModelAssetPackage>({
      capacity: options.capacity,
      lookup: () => Effect.die("EncoderModelAssetCache.get should use getOption"),
      timeToLive: Duration.infinity,
    });

    const get = Effect.fn("EncoderModelAssetCache.get")(function*(packageId: string) {
      const value = yield* EffectCache.getOption(cache, packageId);
      return Option.getOrNull(value);
    });

    const put = Effect.fn("EncoderModelAssetCache.put")(function*(pkg: ModelAssetPackage) {
      yield* EffectCache.set(cache, pkg.key.packageId, pkg);
    });

    const remove = Effect.fn("EncoderModelAssetCache.remove")(function*(packageId: string) {
      yield* EffectCache.invalidate(cache, packageId);
    });

    return EncoderModelAssetCache.of({
      get,
      put,
      remove,
    });
  });
}
