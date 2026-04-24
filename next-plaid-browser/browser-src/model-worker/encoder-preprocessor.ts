import { Context, Effect, Layer, Schema, Scope } from "effect";

import {
  type WorkerRuntimeError,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import { EncoderModelAssets } from "./encoder-model-assets.js";
import type { OnnxConfig } from "./types.js";
import {
  loadBrowserPreprocessWasmModule,
  type BrowserPreprocessWasmModule,
} from "./preprocess-wasm-loader.js";

const decodeSummary = Schema.decodeUnknownEffect(
  Schema.Struct({
    query_length: Schema.Number,
    document_length: Schema.Number,
    uses_token_type_ids: Schema.Boolean,
    mask_token_id: Schema.Number,
    pad_token_id: Schema.Number,
    query_prefix_id: Schema.Number,
    document_prefix_id: Schema.Number,
  }),
);

const decodePreparedInput = Schema.decodeUnknownEffect(
  Schema.Struct({
    input_ids: Schema.Array(Schema.Number),
    attention_mask: Schema.Array(Schema.Number),
    token_type_ids: Schema.optional(Schema.NullOr(Schema.Array(Schema.Number))),
    active_length: Schema.Number,
  }),
);

const decodePreparedDocument = Schema.decodeUnknownEffect(
  Schema.Struct({
    input_ids: Schema.Array(Schema.Number),
    attention_mask: Schema.Array(Schema.Number),
    token_type_ids: Schema.optional(Schema.NullOr(Schema.Array(Schema.Number))),
    retain_row_indices: Schema.Array(Schema.Number),
    active_length: Schema.Number,
  }),
);

export interface PreprocessorSummary {
  readonly query_length: number;
  readonly document_length: number;
  readonly uses_token_type_ids: boolean;
  readonly mask_token_id: number;
  readonly pad_token_id: number;
  readonly query_prefix_id: number;
  readonly document_prefix_id: number;
}

export interface PreparedEncoderInput {
  readonly inputIds: BigInt64Array;
  readonly attentionMask: BigInt64Array;
  readonly tokenTypeIds: BigInt64Array | null;
  readonly inputIdValues: readonly number[];
  readonly attentionMaskValues: readonly number[];
  readonly tokenTypeIdValues: readonly number[] | null;
  readonly activeLength: number;
}

export interface PreparedDocumentInput extends PreparedEncoderInput {
  readonly retainRowIndices: readonly number[];
}

export interface EncoderPreprocessorApi {
  readonly summary: PreprocessorSummary;
  readonly prepareQuery: (
    text: string,
  ) => Effect.Effect<PreparedEncoderInput, WorkerRuntimeError>;
  readonly prepareDocument: (
    text: string,
  ) => Effect.Effect<PreparedDocumentInput, WorkerRuntimeError>;
  readonly reset: () => Effect.Effect<void, WorkerRuntimeError>;
}

export class EncoderPreprocessor
  extends Context.Service<EncoderPreprocessor, EncoderPreprocessorApi>()(
    "next-plaid-browser/EncoderPreprocessor",
  )
{
  static readonly layer = Layer.effect(EncoderPreprocessor)(
    makeEncoderPreprocessor(),
  );
}

function asPreparedSequence(
  operation: string,
  value: {
    readonly input_ids: readonly number[];
    readonly attention_mask: readonly number[];
    readonly token_type_ids?: readonly number[] | null | undefined;
    readonly active_length: number;
  },
  expectedLength: number,
  usesTokenTypeIds: boolean,
): Effect.Effect<PreparedEncoderInput, WorkerRuntimeError> {
  return Effect.gen(function*() {
    const inputIdValues = yield* normalizeIntegerArray(
      `${operation}.input_ids`,
      value.input_ids,
      expectedLength,
    );
    const attentionMaskValues = yield* normalizeIntegerArray(
      `${operation}.attention_mask`,
      value.attention_mask,
      expectedLength,
    );
    const tokenTypeIdValues = value.token_type_ids ?? null;

    if (usesTokenTypeIds) {
      if (tokenTypeIdValues === null) {
        return yield* workerRuntimeError({
          operation,
          message: "preprocessor did not return token_type_ids for a model that requires them",
          details: value,
        });
      }
    } else if (tokenTypeIdValues !== null) {
      return yield* workerRuntimeError({
        operation,
        message: "preprocessor returned token_type_ids for a model that does not use them",
        details: value,
      });
    }

    const normalizedTokenTypeValues = tokenTypeIdValues === null
      ? null
      : yield* normalizeIntegerArray(
        `${operation}.token_type_ids`,
        tokenTypeIdValues,
        expectedLength,
      );
    if (
      !Number.isInteger(value.active_length) ||
      value.active_length < 0 ||
      value.active_length > expectedLength
    ) {
      return yield* workerRuntimeError({
        operation: `${operation}.active_length`,
        message: "preprocessor returned an invalid active_length",
        details: value.active_length,
      });
    }

    return {
      inputIds: toBigInt64Array(inputIdValues),
      attentionMask: toBigInt64Array(attentionMaskValues),
      tokenTypeIds:
        normalizedTokenTypeValues === null
          ? null
          : toBigInt64Array(normalizedTokenTypeValues),
      inputIdValues,
      attentionMaskValues,
      tokenTypeIdValues: normalizedTokenTypeValues,
      activeLength: value.active_length,
    };
  });
}

function normalizeIntegerArray(
  operation: string,
  values: readonly number[],
  expectedLength: number,
): Effect.Effect<number[], WorkerRuntimeError> {
  return Effect.sync(() => {
    if (values.length !== expectedLength) {
      throw workerRuntimeError({
        operation,
        message: `expected length ${expectedLength}, received ${values.length}`,
        details: values,
      });
    }

    return values.map((value, index) => {
      if (!Number.isInteger(value) || value < 0) {
        throw workerRuntimeError({
          operation,
          message: `expected a non-negative integer at index ${index}`,
          details: { value, index },
        });
      }
      return value;
    });
  });
}

function toBigInt64Array(values: readonly number[]): BigInt64Array {
  return BigInt64Array.from(values, (value) => BigInt(value));
}

function validateSummary(
  summary: PreprocessorSummary,
  config: OnnxConfig,
): Effect.Effect<void, WorkerRuntimeError> {
  return Effect.gen(function*() {
    const checks: Array<
      readonly [keyof PreprocessorSummary, number | boolean, number | boolean]
    > = [
      ["query_length", summary.query_length, config.query_length],
      ["document_length", summary.document_length, config.document_length],
      [
        "uses_token_type_ids",
        summary.uses_token_type_ids,
        config.uses_token_type_ids,
      ],
      ["mask_token_id", summary.mask_token_id, config.mask_token_id],
      ["pad_token_id", summary.pad_token_id, config.pad_token_id],
    ];

    for (const [field, actual, expected] of checks) {
      if (actual !== expected) {
        return yield* workerRuntimeError({
          operation: "encoder_preprocessor.validate_summary",
          message: `preprocessor summary mismatch for ${field}`,
          details: { field, actual, expected },
        });
      }
    }

    if (
      config.query_prefix_id !== undefined &&
      summary.query_prefix_id !== config.query_prefix_id
    ) {
      return yield* workerRuntimeError({
        operation: "encoder_preprocessor.validate_query_prefix_id",
        message: "preprocessor query_prefix_id does not match onnx config",
        details: {
          actual: summary.query_prefix_id,
          expected: config.query_prefix_id,
        },
      });
    }

    if (
      config.document_prefix_id !== undefined &&
      summary.document_prefix_id !== config.document_prefix_id
    ) {
      return yield* workerRuntimeError({
        operation: "encoder_preprocessor.validate_document_prefix_id",
        message: "preprocessor document_prefix_id does not match onnx config",
        details: {
          actual: summary.document_prefix_id,
          expected: config.document_prefix_id,
        },
      });
    }
  });
}

function callWasm<T>(
  operation: string,
  f: () => T,
): Effect.Effect<T, WorkerRuntimeError> {
  return Effect.try({
    try: f,
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        operation,
        error,
        "encoder preprocessor wasm call failed",
      ),
  });
}

function decodeWasmSummary(
  value: unknown,
): Effect.Effect<PreprocessorSummary, WorkerRuntimeError> {
  return decodeSummary(value).pipe(
    Effect.mapError((error) =>
      workerRuntimeErrorFromUnknown(
        "encoder_preprocessor.decode_summary",
        error,
        "failed to decode preprocessor summary",
      ),
    ),
  );
}

function prepareQuery(
  module: BrowserPreprocessWasmModule,
  summary: PreprocessorSummary,
  text: string,
): Effect.Effect<PreparedEncoderInput, WorkerRuntimeError> {
  return callWasm("encoder_preprocessor.prepare_query", () =>
    module.prepare_query(text)
  ).pipe(
    Effect.flatMap((value) =>
      decodePreparedInput(value).pipe(
        Effect.mapError((error) =>
          workerRuntimeErrorFromUnknown(
            "encoder_preprocessor.decode_query",
            error,
            "failed to decode prepared query input",
          ),
        ),
      )
    ),
    Effect.flatMap((prepared) =>
      asPreparedSequence(
        "encoder_preprocessor.prepare_query",
        prepared,
        summary.query_length,
        summary.uses_token_type_ids,
      )
    ),
  );
}

function prepareDocument(
  module: BrowserPreprocessWasmModule,
  summary: PreprocessorSummary,
  text: string,
): Effect.Effect<PreparedDocumentInput, WorkerRuntimeError> {
  return callWasm("encoder_preprocessor.prepare_document", () =>
    module.prepare_document(text)
  ).pipe(
    Effect.flatMap((value) =>
      decodePreparedDocument(value).pipe(
        Effect.mapError((error) =>
          workerRuntimeErrorFromUnknown(
            "encoder_preprocessor.decode_document",
            error,
            "failed to decode prepared document input",
          ),
        ),
      )
    ),
    Effect.flatMap((prepared) =>
      Effect.gen(function*() {
        const base = yield* asPreparedSequence(
          "encoder_preprocessor.prepare_document",
          prepared,
          summary.document_length,
          summary.uses_token_type_ids,
        );
        const retainRowIndices = yield* normalizeIntegerArray(
          "encoder_preprocessor.prepare_document.retain_row_indices",
          prepared.retain_row_indices,
          prepared.retain_row_indices.length,
        );
        for (const index of retainRowIndices) {
          if (index >= prepared.active_length) {
            return yield* workerRuntimeError({
              operation: "encoder_preprocessor.prepare_document.retain_row_indices",
              message: "retain_row_indices must refer only to active document rows",
              details: { index, activeLength: prepared.active_length },
            });
          }
        }

        return {
          ...base,
          retainRowIndices,
        };
      })
    ),
  );
}

function makeEncoderPreprocessor(): Effect.Effect<
  EncoderPreprocessorApi,
  WorkerRuntimeError,
  EncoderModelAssets | Scope.Scope
> {
  return Effect.gen(function*() {
    const assets = yield* EncoderModelAssets;

    const runtime = yield* Effect.acquireRelease(
      Effect.gen(function*() {
        const module = yield* loadBrowserPreprocessWasmModule();
        yield* Effect.tryPromise({
          try: () => module.default(),
          catch: (error) =>
            workerRuntimeErrorFromUnknown(
              "encoder_preprocessor.init_module",
              error,
              "failed to initialize encoder preprocessor wasm module",
            ),
        });
        const summary = yield* callWasm("encoder_preprocessor.init", () =>
          module.init(assets.tokenizerBytes, assets.onnxConfigBytes)
        ).pipe(Effect.flatMap(decodeWasmSummary));
        yield* validateSummary(summary, assets.config);
        return { module, summary } as const;
      }),
      ({ module }) =>
        callWasm("encoder_preprocessor.reset", () => module.reset()).pipe(
          Effect.orElseSucceed(() => void 0),
        ),
    );

    return EncoderPreprocessor.of({
      summary: runtime.summary,
      prepareQuery: (text) => prepareQuery(runtime.module, runtime.summary, text),
      prepareDocument: (text) =>
        prepareDocument(runtime.module, runtime.summary, text),
      reset: () => callWasm("encoder_preprocessor.reset", () => runtime.module.reset()),
    });
  });
}
