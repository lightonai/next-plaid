import { Effect, Schema } from "effect";

import type {
  EncoderIdentity,
  QueryEmbeddingsPayload,
} from "./search-contract.js";

export const EmbeddingDtypeSchema = Schema.Literal("f32_le");

export const EmbeddingLayoutSchema = Schema.Union([
  Schema.Literal("ragged"),
  Schema.Literal("padded_query_length"),
]);

export const EncoderIdentitySchema = Schema.Struct({
  encoder_id: Schema.String,
  encoder_build: Schema.String,
  embedding_dim: Schema.Number,
  normalized: Schema.Boolean,
});

const _encoderIdentitySchemaMatchesGenerated:
  [Schema.Schema.Type<typeof EncoderIdentitySchema>] extends [EncoderIdentity] ? true : never =
    true;
void _encoderIdentitySchemaMatchesGenerated;

const QueryEmbeddingRowsSchema = Schema.Array(Schema.Array(Schema.Finite));
const BinaryQueryEmbeddingShapeSchema = Schema.Tuple([Schema.Number, Schema.Number]);

export const QueryEmbeddingsPayloadSchema = Schema.Struct({
  encoder: EncoderIdentitySchema,
  dtype: EmbeddingDtypeSchema,
  layout: EmbeddingLayoutSchema,
  embeddings: Schema.optional(
    Schema.NullOr(QueryEmbeddingRowsSchema),
  ),
  embeddings_b64: Schema.optional(Schema.NullOr(Schema.String)),
  shape: Schema.optional(Schema.NullOr(BinaryQueryEmbeddingShapeSchema)),
});

const _queryEmbeddingsPayloadSchemaMatchesGenerated:
  [QueryEmbeddingsPayload] extends [Schema.Schema.Type<typeof QueryEmbeddingsPayloadSchema>] ? true
    : never = true;
void _queryEmbeddingsPayloadSchemaMatchesGenerated;

export type InlineQueryEmbeddingsPayload = QueryEmbeddingsPayload & {
  embeddings: number[][];
  embeddings_b64?: null;
  shape?: null;
};

export type BinaryQueryEmbeddingsPayload = QueryEmbeddingsPayload & {
  embeddings?: null;
  embeddings_b64: string;
  shape: [number, number];
};

function hasInlineEmbeddingField(
  payload: QueryEmbeddingsPayload,
): payload is QueryEmbeddingsPayload & { embeddings: number[][] } {
  return payload.embeddings !== undefined && payload.embeddings !== null;
}

function hasBinaryEmbeddingField(
  payload: QueryEmbeddingsPayload,
): payload is QueryEmbeddingsPayload & { embeddings_b64: string } {
  return payload.embeddings_b64 !== undefined && payload.embeddings_b64 !== null;
}

function isInlineQueryEmbeddingsPayloadValue(
  payload: QueryEmbeddingsPayload,
): payload is InlineQueryEmbeddingsPayload {
  return hasInlineEmbeddingField(payload) &&
    !hasBinaryEmbeddingField(payload) &&
    (payload.shape === undefined || payload.shape === null);
}

function isBinaryQueryEmbeddingsPayloadValue(
  payload: QueryEmbeddingsPayload,
): payload is BinaryQueryEmbeddingsPayload {
  return !hasInlineEmbeddingField(payload) &&
    hasBinaryEmbeddingField(payload) &&
    payload.shape !== undefined &&
    payload.shape !== null;
}

export const InlineQueryEmbeddingsPayloadSchema = QueryEmbeddingsPayloadSchema.check(
  Schema.makeFilter((payload) => {
    const normalizedPayload = normalizeQueryEmbeddingsPayload(payload);

    if (hasBinaryEmbeddingField(normalizedPayload)) {
      return "inline_query_embeddings_expected";
    }

    if (!hasInlineEmbeddingField(normalizedPayload)) {
      return "missing_query_embeddings";
    }

    const issues: Array<Schema.FilterIssue> = [];

    normalizedPayload.embeddings.forEach((row, rowIndex) => {
      if (row.length !== normalizedPayload.encoder.embedding_dim) {
        issues.push({
          path: ["embeddings", rowIndex],
          issue: "query_embedding_dim_mismatch",
        });
      }
    });

    return issues.length === 0 ? undefined : issues;
  }),
);

export const BinaryQueryEmbeddingsPayloadSchema = QueryEmbeddingsPayloadSchema.check(
  Schema.makeFilter((payload) => {
    const normalizedPayload = normalizeQueryEmbeddingsPayload(payload);

    if (hasInlineEmbeddingField(normalizedPayload)) {
      return "binary_query_embeddings_expected";
    }

    if (!hasBinaryEmbeddingField(normalizedPayload)) {
      return "missing_query_embeddings";
    }

    if (normalizedPayload.shape === undefined || normalizedPayload.shape === null) {
      return {
        path: ["shape"],
        issue: "missing_binary_shape",
      };
    }

    const [, actualDimension] = normalizedPayload.shape;
    return actualDimension === normalizedPayload.encoder.embedding_dim
      ? undefined
      : {
          path: ["shape", 1],
          issue: "query_embedding_dim_mismatch",
        };
  }),
);

export const DenseQueryEmbeddingsPayloadSchema = QueryEmbeddingsPayloadSchema.check(
  Schema.makeFilter((payload) => {
    const normalizedPayload = normalizeQueryEmbeddingsPayload(payload);
    const issues: Array<Schema.FilterIssue> = [];
    const hasInline = hasInlineEmbeddingField(normalizedPayload);
    const hasBinary = hasBinaryEmbeddingField(normalizedPayload);

    if (hasInline && hasBinary) {
      return "ambiguous_query_embeddings";
    }

    if (!hasInline && !hasBinary) {
      return "missing_query_embeddings";
    }

    if (hasBinary && (payload.shape === undefined || payload.shape === null)) {
      return {
        path: ["shape"],
        issue: "missing_binary_shape",
      };
    }

    const expectedDimension = normalizedPayload.encoder.embedding_dim;

    if (hasInline) {
      normalizedPayload.embeddings.forEach((row, rowIndex) => {
        if (row.length !== expectedDimension) {
          issues.push({
            path: ["embeddings", rowIndex],
            issue: "query_embedding_dim_mismatch",
          });
        }
      });
    }

    if (hasBinary && normalizedPayload.shape !== undefined && normalizedPayload.shape !== null) {
      const [, actualDimension] = normalizedPayload.shape;
      if (actualDimension !== expectedDimension) {
        issues.push({
          path: ["shape", 1],
          issue: "query_embedding_dim_mismatch",
        });
      }
    }

    return issues.length === 0 ? undefined : issues;
  }),
);

export function normalizeQueryEmbeddingsPayload(
  payload: Schema.Schema.Type<typeof QueryEmbeddingsPayloadSchema>,
): QueryEmbeddingsPayload {
  const normalized: QueryEmbeddingsPayload = {
    encoder: payload.encoder,
    dtype: payload.dtype,
    layout: payload.layout,
  };

  if (payload.embeddings !== undefined) {
    normalized.embeddings = payload.embeddings === null
      ? null
      : payload.embeddings.map((row) => [...row]);
  }

  if (payload.embeddings_b64 !== undefined) {
    normalized.embeddings_b64 = payload.embeddings_b64;
  }

  if (payload.shape !== undefined) {
    normalized.shape = payload.shape === null
      ? null
      : [payload.shape[0], payload.shape[1]];
  }

  return normalized;
}

function makeSchemaGuard<T>(
  predicate: (value: unknown) => boolean,
): (value: unknown) => value is T {
  return (value: unknown): value is T => predicate(value);
}

export const isEncoderIdentity = makeSchemaGuard<EncoderIdentity>(
  Schema.is(EncoderIdentitySchema),
);

export const isQueryEmbeddingsPayload = makeSchemaGuard<QueryEmbeddingsPayload>(
  Schema.is(QueryEmbeddingsPayloadSchema),
);

export const isInlineQueryEmbeddingsPayload = (
  value: unknown,
): value is InlineQueryEmbeddingsPayload =>
  isQueryEmbeddingsPayload(value) && isInlineQueryEmbeddingsPayloadValue(value);

export const isBinaryQueryEmbeddingsPayload = (
  value: unknown,
): value is BinaryQueryEmbeddingsPayload =>
  isQueryEmbeddingsPayload(value) && isBinaryQueryEmbeddingsPayloadValue(value);

export const isDenseQueryEmbeddingsPayload = makeSchemaGuard<QueryEmbeddingsPayload>(
  Schema.is(DenseQueryEmbeddingsPayloadSchema),
);

export const decodeQueryEmbeddingsPayload = (
  value: unknown,
): Effect.Effect<QueryEmbeddingsPayload, Schema.SchemaError> =>
  Schema.decodeUnknownEffect(QueryEmbeddingsPayloadSchema)(value).pipe(
    Effect.map(normalizeQueryEmbeddingsPayload),
  );

export const decodeInlineQueryEmbeddingsPayload = (
  value: unknown,
): Effect.Effect<InlineQueryEmbeddingsPayload, Schema.SchemaError> =>
  Schema.decodeUnknownEffect(InlineQueryEmbeddingsPayloadSchema)(value).pipe(
    Effect.map(normalizeQueryEmbeddingsPayload),
    Effect.map((payload) => payload as InlineQueryEmbeddingsPayload),
  );

export const decodeBinaryQueryEmbeddingsPayload = (
  value: unknown,
): Effect.Effect<BinaryQueryEmbeddingsPayload, Schema.SchemaError> =>
  Schema.decodeUnknownEffect(BinaryQueryEmbeddingsPayloadSchema)(value).pipe(
    Effect.map(normalizeQueryEmbeddingsPayload),
    Effect.map((payload) => payload as BinaryQueryEmbeddingsPayload),
  );

export const decodeDenseQueryEmbeddingsPayload = (
  value: unknown,
): Effect.Effect<QueryEmbeddingsPayload, Schema.SchemaError> =>
  Schema.decodeUnknownEffect(DenseQueryEmbeddingsPayloadSchema)(value).pipe(
    Effect.map(normalizeQueryEmbeddingsPayload),
  );
