import { describe, expect, it } from "@effect/vitest";
import { Context, Effect, Layer, Scope } from "effect";
import * as Exit from "effect/Exit";
import { vi } from "vitest";

import type { ResolvedEncoderModelAssets } from "./encoder-model-assets.js";
import { EncoderModelAssets } from "./encoder-model-assets.js";
import { EncoderPreprocessor } from "./encoder-preprocessor.js";

interface MockPreprocessModule {
  readonly default: () => Promise<unknown>;
  readonly init: (
    tokenizerJsonBytes: Uint8Array,
    onnxConfigJsonBytes: Uint8Array,
  ) => unknown;
  readonly prepare_query: (text: string) => unknown;
  readonly prepare_document: (text: string) => unknown;
  readonly reset: () => void;
}

const loaderState: {
  module: MockPreprocessModule | null;
} = {
  module: null,
};

vi.mock("./preprocess-wasm-loader.js", () => ({
  loadBrowserPreprocessWasmModule: () =>
    Effect.sync(() => {
      if (loaderState.module === null) {
        throw new Error("mock preprocess module not configured");
      }
      return loaderState.module;
    }),
}));

function resolvedAssets(
  overrides?: Partial<ResolvedEncoderModelAssets>,
): ResolvedEncoderModelAssets {
  return {
    modelBytes: new Uint8Array([1, 2, 3]),
    tokenizerBytes: new Uint8Array([4, 5, 6]),
    onnxConfigBytes: new TextEncoder().encode(
      JSON.stringify({
        query_prefix: "[unused0]",
        document_prefix: "[unused1]",
        query_length: 4,
        document_length: 6,
        do_query_expansion: false,
        embedding_dim: 4,
        uses_token_type_ids: false,
        mask_token_id: 4,
        pad_token_id: 0,
        query_prefix_id: 5,
        document_prefix_id: 6,
        skiplist_words: [],
        do_lower_case: true,
      }),
    ),
    config: {
      query_prefix: "[unused0]",
      document_prefix: "[unused1]",
      query_length: 4,
      document_length: 6,
      do_query_expansion: false,
      embedding_dim: 4,
      uses_token_type_ids: false,
      mask_token_id: 4,
      pad_token_id: 0,
      query_prefix_id: 5,
      document_prefix_id: 6,
      skiplist_words: [],
      do_lower_case: true,
    },
    persistentStorage: true,
    ...overrides,
  };
}

function mockModule(
  overrides?: Partial<MockPreprocessModule>,
): {
  readonly module: MockPreprocessModule;
  readonly resetCalls: { current: number };
} {
  const resetCalls = { current: 0 };
  const module: MockPreprocessModule = {
    default: () => Promise.resolve(undefined),
    init: () => ({
      query_length: 4,
      document_length: 6,
      uses_token_type_ids: false,
      mask_token_id: 4,
      pad_token_id: 0,
      query_prefix_id: 5,
      document_prefix_id: 6,
    }),
    prepare_query: () => ({
      input_ids: [2, 5, 7, 3],
      attention_mask: [1, 1, 1, 1],
      token_type_ids: undefined,
      active_length: 4,
    }),
    prepare_document: () => ({
      input_ids: [2, 6, 7, 8, 3, 0],
      attention_mask: [1, 1, 1, 1, 1, 0],
      token_type_ids: undefined,
      retain_row_indices: [0, 2, 4],
      active_length: 5,
    }),
    reset: () => {
      resetCalls.current += 1;
    },
    ...overrides,
  };

  return { module, resetCalls };
}

function buildPreprocessor(
  scope: Scope.Scope,
  assets: ResolvedEncoderModelAssets,
) {
  return Layer.buildWithScope(
    Layer.fresh(EncoderPreprocessor.layer).pipe(
      Layer.provide(
        Layer.succeed(EncoderModelAssets)(
          EncoderModelAssets.of(assets),
        ),
      ),
    ),
    scope,
  ).pipe(Effect.map((context) => Context.get(context, EncoderPreprocessor)));
}

describe("EncoderPreprocessor", () => {
  it.effect("fails init when the wasm summary disagrees with the parsed onnx config", () =>
    Effect.gen(function*() {
      const { module } = mockModule({
        init: () => ({
          query_length: 8,
          document_length: 6,
          uses_token_type_ids: false,
          mask_token_id: 4,
          pad_token_id: 0,
          query_prefix_id: 5,
          document_prefix_id: 6,
        }),
      });
      loaderState.module = module;

      const scope = yield* Scope.make();
      yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

      const exit = yield* Effect.exit(
        buildPreprocessor(scope, resolvedAssets()),
      );

      expect(exit._tag).toBe("Failure");
    }),
  );

  it.effect("normalizes prepared document rows into the worker input shape", () =>
    Effect.gen(function*() {
      const { module } = mockModule();
      loaderState.module = module;

      const scope = yield* Scope.make();
      yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

      const preprocessor = yield* buildPreprocessor(scope, resolvedAssets());
      const prepared = yield* preprocessor.prepareDocument("alpha beta");

      expect(Array.from(prepared.inputIds)).toEqual([2n, 6n, 7n, 8n, 3n, 0n]);
      expect(Array.from(prepared.attentionMask)).toEqual([1n, 1n, 1n, 1n, 1n, 0n]);
      expect(prepared.tokenTypeIds).toBeNull();
      expect(prepared.inputIdValues).toEqual([2, 6, 7, 8, 3, 0]);
      expect(prepared.retainRowIndices).toEqual([0, 2, 4]);
      expect(prepared.activeLength).toBe(5);
    }),
  );

  it.effect("normalizes prepared query active length into the worker input shape", () =>
    Effect.gen(function*() {
      const { module } = mockModule();
      loaderState.module = module;

      const scope = yield* Scope.make();
      yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));

      const preprocessor = yield* buildPreprocessor(scope, resolvedAssets());
      const prepared = yield* preprocessor.prepareQuery("alpha");

      expect(Array.from(prepared.inputIds)).toEqual([2n, 5n, 7n, 3n]);
      expect(prepared.activeLength).toBe(4);
    }),
  );

  it.effect("resets the wasm preprocessor when the scope closes", () =>
    Effect.gen(function*() {
      const { module, resetCalls } = mockModule();
      loaderState.module = module;

      const scope = yield* Scope.make();
      const preprocessor = yield* buildPreprocessor(scope, resolvedAssets());
      yield* preprocessor.prepareQuery("alpha");
      yield* Scope.close(scope, Exit.void);

      expect(resetCalls.current).toBe(1);
    }),
  );
});
