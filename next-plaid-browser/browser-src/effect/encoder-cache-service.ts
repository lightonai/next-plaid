import { Cache as EffectCache, Context, Duration, Effect, Layer } from "effect";
import * as Exit from "effect/Exit";

import type { EncodedQuery } from "../model-worker/types.js";
import type { EncoderClientError } from "./client-errors.js";

const DEFAULT_QUERY_CACHE_CAPACITY = 256;

export interface EncoderCacheServiceApi {
  readonly get: (key: string) => Effect.Effect<EncodedQuery, EncoderClientError>;
  readonly clear: () => Effect.Effect<void>;
}

export class EncoderCacheService
  extends Context.Service<EncoderCacheService, EncoderCacheServiceApi>()(
    "next-plaid-browser/EncoderCacheService",
  )
{
  static layer = (options: {
    readonly lookup: (key: string) => Effect.Effect<EncodedQuery, EncoderClientError>;
    readonly capacity?: number | undefined;
  }) =>
    Layer.effect(EncoderCacheService)(
      EffectCache.makeWith<string, EncodedQuery, EncoderClientError>(
        options.lookup,
        {
          capacity: options.capacity ?? DEFAULT_QUERY_CACHE_CAPACITY,
          timeToLive: (exit) =>
            Exit.isSuccess(exit) ? Duration.infinity : Duration.zero,
        },
      ).pipe(
        Effect.map((cache) =>
          EncoderCacheService.of({
            get: Effect.fn("EncoderCacheService.get")(function*(key: string) {
              return yield* EffectCache.get(cache, key);
            }),
            clear: Effect.fn("EncoderCacheService.clear")(function*() {
              yield* EffectCache.invalidateAll(cache);
            }),
          }),
        ),
      ),
    );
}
