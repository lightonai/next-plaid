import { Context, Effect, Layer, Schema, SchemaIssue, Stream, SubscriptionRef } from "effect";

import type { EncoderCapabilities } from "../model-worker/types.js";
import type {
  EncoderLifecycleEvent,
  EncoderStateSnapshot,
} from "./encoder-worker-client.js";
import { EncoderWorkerClient } from "./encoder-worker-client.js";
import type {
  EncoderIdentity,
  QueryEmbeddingsPayload,
  SearchRequestEnvelope,
  SearchResultsResponseEnvelope,
} from "../shared/search-contract.js";
import {
  decodeDenseQueryEmbeddingsPayload,
  isBinaryQueryEmbeddingsPayload,
  isInlineQueryEmbeddingsPayload,
} from "../shared/search-contract-schema.js";
import {
  type LoadedSearchIndexMetadata,
  SearchWorkerClient,
  type SearchWorkerState,
} from "./search-worker-client.js";
import {
  type BrowserRuntimeError,
  permanentClientError,
} from "./client-errors.js";

export interface EncodeAndSearchArgs {
  readonly text: string;
  readonly requestId?: string | undefined;
  readonly searchRequest: {
    readonly type: SearchRequestEnvelope["type"];
    readonly name: SearchRequestEnvelope["name"];
    readonly request: Omit<SearchRequestEnvelope["request"], "queries">;
  };
}

export interface BrowserSearchRuntimeApi {
  readonly encoderState: SubscriptionRef.SubscriptionRef<EncoderStateSnapshot>;
  readonly searchState: SubscriptionRef.SubscriptionRef<SearchWorkerState>;
  readonly encoderEvents: Stream.Stream<EncoderLifecycleEvent, never>;
  readonly searchWithEmbeddings: (
    request: SearchRequestEnvelope,
  ) => Effect.Effect<SearchResultsResponseEnvelope, BrowserRuntimeError>;
  readonly encodeAndSearch: (
    args: EncodeAndSearchArgs,
  ) => Effect.Effect<SearchResultsResponseEnvelope, BrowserRuntimeError>;
}

export class BrowserSearchRuntime
  extends Context.Service<BrowserSearchRuntime, BrowserSearchRuntimeApi>()(
    "next-plaid-browser/BrowserSearchRuntime",
  )
{
  static layer = (): Layer.Layer<
    BrowserSearchRuntime,
    never,
    SearchWorkerClient | EncoderWorkerClient
  > => Layer.effect(BrowserSearchRuntime)(makeBrowserSearchRuntime);
}

function encoderIdentityFromCapabilities(
  capabilities: EncoderCapabilities,
): EncoderIdentity {
  return {
    encoder_id: capabilities.encoderId,
    encoder_build: capabilities.encoderBuild,
    embedding_dim: capabilities.embeddingDim,
    normalized: capabilities.normalized,
  };
}

function sameEncoderIdentity(
  left: EncoderIdentity,
  right: EncoderIdentity,
): boolean {
  return (
    left.encoder_id === right.encoder_id &&
    left.encoder_build === right.encoder_build &&
    left.embedding_dim === right.embedding_dim &&
    left.normalized === right.normalized
  );
}

function missingLoadedIndexError(
  indexName: string,
  operation: string,
): BrowserRuntimeError {
  return permanentClientError({
    cause: "index_not_loaded",
    message: `search index "${indexName}" is not loaded in the current browser runtime scope`,
    operation,
    details: { indexName },
  });
}

function incompatibleEncoderError(options: {
  readonly operation: string;
  readonly indexName: string;
  readonly expected: EncoderIdentity;
  readonly actual: EncoderIdentity;
}): BrowserRuntimeError {
  return permanentClientError({
    cause: "encoder_identity_mismatch",
    message: `query encoder does not match loaded index "${options.indexName}"`,
    operation: options.operation,
    details: {
      indexName: options.indexName,
      expected: options.expected,
      actual: options.actual,
    },
  });
}

function queryDimensionMismatchError(options: {
  readonly operation: string;
  readonly indexName: string;
  readonly expectedDimension: number;
  readonly actualDimension: number;
  readonly queryIndex: number;
}): BrowserRuntimeError {
  return permanentClientError({
    cause: "query_embedding_dim_mismatch",
    message: `query embedding dimension does not match loaded index "${options.indexName}"`,
    operation: options.operation,
    details: {
      indexName: options.indexName,
      queryIndex: options.queryIndex,
      expectedDimension: options.expectedDimension,
      actualDimension: options.actualDimension,
    },
  });
}

function malformedQueryPayloadError(options: {
  readonly operation: string;
  readonly indexName: string;
  readonly queryIndex: number;
  readonly cause: string;
  readonly message: string;
  readonly payload: unknown;
  readonly schemaIssues?: unknown;
}): BrowserRuntimeError {
  return permanentClientError({
    cause: options.cause,
    message: `${options.message} for loaded index "${options.indexName}"`,
    operation: options.operation,
    details: {
      indexName: options.indexName,
      queryIndex: options.queryIndex,
      payload: options.payload,
      schemaIssues: options.schemaIssues ?? null,
    },
  });
}

const queryPayloadIssueFormatter = SchemaIssue.makeFormatterStandardSchemaV1();

type QueryPayloadValidationCause =
  | "ambiguous_query_embeddings"
  | "invalid_query_embeddings"
  | "missing_binary_shape"
  | "missing_query_embeddings"
  | "query_embedding_dim_mismatch";

function queryPayloadValidationCause(
  issues: ReadonlyArray<{ readonly message: string }>,
): QueryPayloadValidationCause {
  switch (issues[0]?.message) {
    case "ambiguous_query_embeddings":
      return "ambiguous_query_embeddings";
    case "missing_binary_shape":
      return "missing_binary_shape";
    case "missing_query_embeddings":
      return "missing_query_embeddings";
    case "query_embedding_dim_mismatch":
      return "query_embedding_dim_mismatch";
    default:
      return "invalid_query_embeddings";
  }
}

function queryPayloadValidationMessage(
  cause: QueryPayloadValidationCause,
): string {
  switch (cause) {
    case "ambiguous_query_embeddings":
      return "query payload includes both inline and binary embeddings";
    case "missing_binary_shape":
      return "binary query embeddings are missing shape metadata";
    case "missing_query_embeddings":
      return "query payload does not contain embeddings";
    case "query_embedding_dim_mismatch":
      return "query embedding payload does not match its encoder identity";
    case "invalid_query_embeddings":
      return "query payload failed schema validation";
  }
}

function validatedQueryEmbeddingDimension(
  payload: QueryEmbeddingsPayload,
): number {
  if (isInlineQueryEmbeddingsPayload(payload)) {
    return payload.encoder.embedding_dim;
  }

  if (isBinaryQueryEmbeddingsPayload(payload)) {
    return payload.shape[1];
  }

  return payload.encoder.embedding_dim;
}

function validateDenseQueryPayload(
  payload: QueryEmbeddingsPayload,
  options: {
    readonly operation: string;
    readonly indexName: string;
    readonly queryIndex: number;
  },
): Effect.Effect<QueryEmbeddingsPayload, BrowserRuntimeError> {
  return decodeDenseQueryEmbeddingsPayload(payload).pipe(
    Effect.mapError((error: Schema.SchemaError) => {
      const issues = queryPayloadIssueFormatter(error.issue).issues;
      const cause = queryPayloadValidationCause(issues);
      return malformedQueryPayloadError({
        operation: options.operation,
        indexName: options.indexName,
        queryIndex: options.queryIndex,
        cause,
        message: queryPayloadValidationMessage(cause),
        payload,
        schemaIssues: issues,
      });
    }),
  );
}

function ensureQueryCompatibleWithIndex(
  indexMetadata: LoadedSearchIndexMetadata,
  payload: QueryEmbeddingsPayload,
  queryIndex: number,
  operation: string,
): Effect.Effect<void, BrowserRuntimeError> {
  return Effect.gen(function*() {
    const validatedPayload = yield* validateDenseQueryPayload(payload, {
      operation,
      indexName: indexMetadata.name,
      queryIndex,
    });
    const payloadDimension = validatedQueryEmbeddingDimension(validatedPayload);

    if (payloadDimension !== indexMetadata.summary.dimension) {
      return yield* queryDimensionMismatchError({
        operation,
        indexName: indexMetadata.name,
        queryIndex,
        expectedDimension: indexMetadata.summary.dimension,
        actualDimension: payloadDimension,
      });
    }

    if (
      indexMetadata.encoder !== null &&
      !sameEncoderIdentity(validatedPayload.encoder, indexMetadata.encoder)
    ) {
      return yield* incompatibleEncoderError({
        operation,
        indexName: indexMetadata.name,
        expected: indexMetadata.encoder,
        actual: validatedPayload.encoder,
      });
    }
  });
}

function ensureEncoderCompatibleWithIndex(
  indexMetadata: LoadedSearchIndexMetadata,
  encoder: EncoderIdentity,
  operation: string,
): Effect.Effect<void, BrowserRuntimeError> {
  if (encoder.embedding_dim !== indexMetadata.summary.dimension) {
    return Effect.fail(
      queryDimensionMismatchError({
        operation,
        indexName: indexMetadata.name,
        queryIndex: 0,
        expectedDimension: indexMetadata.summary.dimension,
        actualDimension: encoder.embedding_dim,
      }),
    );
  }

  if (indexMetadata.encoder !== null && !sameEncoderIdentity(encoder, indexMetadata.encoder)) {
    return Effect.fail(
      incompatibleEncoderError({
        operation,
        indexName: indexMetadata.name,
        expected: indexMetadata.encoder,
        actual: encoder,
      }),
    );
  }

  return Effect.void;
}

function loadedIndexMetadata(
  loadedIndices: SubscriptionRef.SubscriptionRef<
    ReadonlyMap<string, LoadedSearchIndexMetadata>
  >,
  indexName: string,
  operation: string,
): Effect.Effect<LoadedSearchIndexMetadata, BrowserRuntimeError> {
  return SubscriptionRef.get(loadedIndices).pipe(
    Effect.flatMap((current) => {
      const metadata = current.get(indexName);
      return metadata === undefined
        ? Effect.fail(missingLoadedIndexError(indexName, operation))
        : Effect.succeed(metadata);
    }),
  );
}

function hasKeywordText(
  request: EncodeAndSearchArgs["searchRequest"]["request"],
): boolean {
  return request.text_query !== undefined &&
    request.text_query !== null &&
    request.text_query.length > 0;
}

function searchMode(
  request: SearchRequestEnvelope["request"],
): "keyword" | "dense" | "hybrid" {
  const hasDense = request.queries !== undefined &&
    request.queries !== null &&
    request.queries.length > 0;
  const hasKeyword = request.text_query !== undefined &&
    request.text_query !== null &&
    request.text_query.length > 0;

  if (hasDense && hasKeyword) {
    return "hybrid";
  }
  if (hasDense) {
    return "dense";
  }
  return "keyword";
}

function keywordOnlyRequest(
  args: EncodeAndSearchArgs,
): SearchRequestEnvelope {
  return {
    type: args.searchRequest.type,
    name: args.searchRequest.name,
    request: {
      ...args.searchRequest.request,
      queries: null,
      alpha: null,
      fusion: null,
    },
  };
}

function validateSearchRequest(
  indexMetadata: LoadedSearchIndexMetadata,
  request: SearchRequestEnvelope,
  operation: string,
): Effect.Effect<void, BrowserRuntimeError> {
  const queries = request.request.queries;
  if (queries === undefined || queries === null) {
    return Effect.void;
  }

  return Effect.forEach(queries, (query, queryIndex) =>
    ensureQueryCompatibleWithIndex(indexMetadata, query, queryIndex, operation), {
    concurrency: "unbounded",
    discard: true,
  });
}

export const makeBrowserSearchRuntime: Effect.Effect<
  BrowserSearchRuntimeApi,
  never,
  SearchWorkerClient | EncoderWorkerClient
> = Effect.gen(function*() {
  const searchClient = yield* SearchWorkerClient;
  const encoderClient = yield* EncoderWorkerClient;

  const searchWithEmbeddings = Effect.fn("BrowserSearchRuntime.searchWithEmbeddings")(
    (request: SearchRequestEnvelope) =>
      loadedIndexMetadata(
        searchClient.loadedIndices,
        request.name,
        "browser_runtime.search_with_embeddings",
      ).pipe(
        Effect.flatMap((indexMetadata) =>
          validateSearchRequest(
            indexMetadata,
            request,
            "browser_runtime.search_with_embeddings",
          ),
        ),
        Effect.andThen(searchClient.search(request)),
        Effect.withLogSpan("browser_runtime.search_with_embeddings"),
        Effect.annotateLogs({
          operation: "browser_runtime.search_with_embeddings",
          index_name: request.name,
          query_count: request.request.queries?.length ?? 0,
          search_mode: searchMode(request.request),
        }),
      ),
  );

  const encodeAndSearch = Effect.fn("BrowserSearchRuntime.encodeAndSearch")(
    (args: EncodeAndSearchArgs) =>
      Effect.gen(function*() {
        const indexMetadata = yield* loadedIndexMetadata(
          searchClient.loadedIndices,
          args.searchRequest.name,
          "browser_runtime.encode_and_search",
        );
        const canFallbackToKeyword = hasKeywordText(args.searchRequest.request);
        const fallbackRequest = keywordOnlyRequest(args);
        const fallbackToKeyword = () =>
          searchClient.search(fallbackRequest).pipe(
            Effect.withLogSpan("browser_runtime.encode_and_search.keyword_fallback"),
            Effect.annotateLogs({
              operation: "browser_runtime.encode_and_search",
              index_name: args.searchRequest.name,
              fallback_mode: "keyword_only",
              search_mode: "keyword",
            }),
          );
        const encoderSnapshot = yield* SubscriptionRef.get(encoderClient.state);

        if (encoderSnapshot.status !== "ready") {
          if (canFallbackToKeyword) {
            return yield* fallbackToKeyword();
          }
        } else {
          const compatibilityResult = yield* Effect.result(
            ensureEncoderCompatibleWithIndex(
              indexMetadata,
              encoderIdentityFromCapabilities(encoderSnapshot.capabilities),
              "browser_runtime.encode_and_search",
            ),
          );
          if (compatibilityResult._tag === "Failure") {
            if (canFallbackToKeyword) {
              return yield* fallbackToKeyword();
            }
            return yield* compatibilityResult.failure;
          }
        }

        const encodedResult = yield* Effect.result(
          encoderClient.encode(
            args.requestId === undefined
              ? { text: args.text }
              : { text: args.text, requestId: args.requestId },
          ),
        );
        if (encodedResult._tag === "Failure") {
          if (canFallbackToKeyword) {
            return yield* fallbackToKeyword();
          }
          return yield* encodedResult.failure;
        }
        const encoded = encodedResult.success;

        const request: SearchRequestEnvelope = {
          type: args.searchRequest.type,
          name: args.searchRequest.name,
          request: {
            ...args.searchRequest.request,
            queries: [encoded.payload],
          },
        };

        return yield* searchWithEmbeddings(request);
      }).pipe(
        Effect.withLogSpan("browser_runtime.encode_and_search"),
        Effect.annotateLogs({
          operation: "browser_runtime.encode_and_search",
          index_name: args.searchRequest.name,
          query_char_len: args.text.length,
          search_mode: hasKeywordText(args.searchRequest.request) ? "hybrid" : "dense",
        }),
      ),
  );

  return {
    encoderState: encoderClient.state,
    searchState: searchClient.state,
    encoderEvents: encoderClient.events,
    searchWithEmbeddings,
    encodeAndSearch,
  } satisfies BrowserSearchRuntimeApi;
});
