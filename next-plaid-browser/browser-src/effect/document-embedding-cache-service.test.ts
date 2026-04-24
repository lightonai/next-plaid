import { expect, it } from "@effect/vitest";
import {
  Context,
  Effect,
  Equal,
  Hash,
  Layer,
  Ref,
  Stream,
  SubscriptionRef,
} from "effect";

import type {
  EncodedDocument,
  EncoderCapabilities,
} from "../model-worker/types.js";
import type { MatrixPayload } from "../generated/MatrixPayload.js";
import { DocumentEmbeddingCacheService, DocumentEmbeddingCacheKey } from "./document-embedding-cache-service.js";
import { DocumentTextDigestService } from "./document-text-digest-service.js";
import type { EncoderStateSnapshot } from "./encoder-worker-client.js";
import { EncoderWorkerClient } from "./encoder-worker-client.js";

interface DocumentEmbeddingCacheHarnessApi {
  readonly calls: Ref.Ref<ReadonlyArray<string>>;
}

class DocumentEmbeddingCacheHarness
  extends Context.Service<DocumentEmbeddingCacheHarness, DocumentEmbeddingCacheHarnessApi>()(
    "next-plaid-browser/tests/DocumentEmbeddingCacheHarness",
  )
{}

function encoderCapabilities(
  overrides: Partial<EncoderCapabilities> = {},
): EncoderCapabilities {
  return {
    backend: "wasm",
    threaded: false,
    persistentStorage: true,
    encoderId: overrides.encoderId ?? "proof-encoder",
    encoderBuild: overrides.encoderBuild ?? "proof-build-1",
    embeddingDim: overrides.embeddingDim ?? 4,
    queryLength: overrides.queryLength ?? 8,
    documentLength: overrides.documentLength ?? 16,
    doQueryExpansion: overrides.doQueryExpansion ?? false,
    usesTokenTypeIds: overrides.usesTokenTypeIds ?? false,
    doLowerCase: overrides.doLowerCase ?? false,
    queryOutputLayout: overrides.queryOutputLayout ?? "ragged",
    normalized: overrides.normalized ?? true,
  };
}

function encodedDocumentForText(
  text: string,
  overrides: Partial<MatrixPayload> = {},
): EncodedDocument {
  return {
    payload: {
      values: overrides.values ?? [text.length, 0.2, 0.3, 0.4],
      rows: overrides.rows ?? 1,
      dim: overrides.dim ?? 4,
    },
    timing: {
      total_ms: 2,
      tokenize_ms: 1,
      inference_ms: 1,
    },
    input_ids: [201],
    attention_mask: [1],
  };
}

function makeDocumentEmbeddingCacheTestLayer(options: {
  readonly capacity?: number | undefined;
  readonly encodeDocument?: ((text: string) => EncodedDocument) | undefined;
} = {}): Layer.Layer<
  DocumentEmbeddingCacheHarness | DocumentEmbeddingCacheService,
  never
> {
  const harnessLayer = Layer.effect(DocumentEmbeddingCacheHarness)(
    Effect.gen(function*() {
      return DocumentEmbeddingCacheHarness.of({
        calls: yield* Ref.make<ReadonlyArray<string>>([]),
      });
    }),
  );

  const encoderLayer = Layer.effect(EncoderWorkerClient)(
    Effect.gen(function*() {
      const harness = yield* DocumentEmbeddingCacheHarness;
      const state = yield* SubscriptionRef.make<EncoderStateSnapshot>({
        status: "ready",
        capabilities: encoderCapabilities(),
        lastError: null,
      });

      return EncoderWorkerClient.of({
        state,
        events: Stream.empty,
        init: () => Effect.die("unused init in document cache test"),
        encodeQuery: () => Effect.die("unused encodeQuery in document cache test"),
        encodeDocument: ({ text }) =>
          Ref.update(harness.calls, (current) => [...current, text]).pipe(
            Effect.as(
              options.encodeDocument?.(text) ?? encodedDocumentForText(text),
            ),
          ),
      });
    }),
  ).pipe(Layer.provide(harnessLayer));

  const cacheLayer = DocumentEmbeddingCacheService.layer({
    capacity: options.capacity,
  }).pipe(
    Layer.provide(Layer.mergeAll(encoderLayer, DocumentTextDigestService.layer)),
  );

  return Layer.mergeAll(harnessLayer, cacheLayer);
}

it.effect("computes the full SHA-256 hex digest for document text", () =>
  Effect.gen(function*() {
    const digestService = yield* DocumentTextDigestService;
    const digest = yield* digestService.sha256Hex("alpha semantic body");

    expect(digest).toBe(
      "51de0f0f7f93c5d6abaf81bc7fa5a0e34f5e41d5a24d30d5ee9fc77659f72a84",
    );
  }).pipe(Effect.provide(DocumentTextDigestService.layer)),
);

it.effect("treats keys with the same encoder identity and digest as equal", () =>
  Effect.sync(() => {
    const left = new DocumentEmbeddingCacheKey({
      encoderId: "proof-encoder",
      encoderBuild: "proof-build-1",
      embeddingDim: 4,
      normalized: true,
      textDigest: "digest-1",
      text: "alpha semantic body",
    });
    const right = new DocumentEmbeddingCacheKey({
      encoderId: "proof-encoder",
      encoderBuild: "proof-build-1",
      embeddingDim: 4,
      normalized: true,
      textDigest: "digest-1",
      text: "different raw text",
    });

    expect(Equal.equals(left, right)).toBe(true);
    expect(Hash.hash(left)).toBe(Hash.hash(right));
  }),
);

it.effect("reuses the cached payload for the same encoder identity and text", () =>
  Effect.gen(function*() {
    const harness = yield* DocumentEmbeddingCacheHarness;
    const cache = yield* DocumentEmbeddingCacheService;
    const capabilities = encoderCapabilities();

    const first = yield* cache.get({
      capabilities,
      text: "alpha semantic body",
    });
    const second = yield* cache.get({
      capabilities,
      text: "alpha semantic body",
    });

    expect(first).toEqual(second);
    expect(yield* Ref.get(harness.calls)).toEqual(["alpha semantic body"]);
  }).pipe(Effect.provide(makeDocumentEmbeddingCacheTestLayer())),
);

it.effect("misses the cache when the encoder identity changes", () =>
  Effect.gen(function*() {
    const harness = yield* DocumentEmbeddingCacheHarness;
    const cache = yield* DocumentEmbeddingCacheService;

    yield* cache.get({
      capabilities: encoderCapabilities(),
      text: "alpha semantic body",
    });
    yield* cache.get({
      capabilities: encoderCapabilities({
        encoderId: "alternate-encoder",
        encoderBuild: "alternate-build-1",
      }),
      text: "alpha semantic body",
    });

    expect(yield* Ref.get(harness.calls)).toEqual([
      "alpha semantic body",
      "alpha semantic body",
    ]);
  }).pipe(Effect.provide(makeDocumentEmbeddingCacheTestLayer())),
);

it.effect("does not cache invalid payloads and retries the encoder on the next lookup", () =>
  Effect.gen(function*() {
    const harness = yield* DocumentEmbeddingCacheHarness;
    const cache = yield* DocumentEmbeddingCacheService;
    const capabilities = encoderCapabilities();

    const first = yield* Effect.result(
      cache.get({
        capabilities,
        text: "alpha semantic body",
      }),
    );
    const second = yield* Effect.result(
      cache.get({
        capabilities,
        text: "alpha semantic body",
      }),
    );

    expect(first._tag).toBe("Failure");
    expect(second._tag).toBe("Failure");
    expect(yield* Ref.get(harness.calls)).toEqual([
      "alpha semantic body",
      "alpha semantic body",
    ]);
  }).pipe(
    Effect.provide(
      makeDocumentEmbeddingCacheTestLayer({
        encodeDocument: (text) =>
          encodedDocumentForText(text, {
            values: [],
            rows: 0,
            dim: 0,
          }),
      }),
    ),
  ),
);

it.effect("evicts older entries when capacity is exceeded", () =>
  Effect.gen(function*() {
    const harness = yield* DocumentEmbeddingCacheHarness;
    const cache = yield* DocumentEmbeddingCacheService;
    const capabilities = encoderCapabilities();

    yield* cache.get({
      capabilities,
      text: "alpha semantic body",
    });
    yield* cache.get({
      capabilities,
      text: "beta semantic body",
    });
    yield* cache.get({
      capabilities,
      text: "alpha semantic body",
    });

    expect(yield* Ref.get(harness.calls)).toEqual([
      "alpha semantic body",
      "beta semantic body",
      "alpha semantic body",
    ]);
  }).pipe(
    Effect.provide(makeDocumentEmbeddingCacheTestLayer({ capacity: 1 })),
  ),
);
