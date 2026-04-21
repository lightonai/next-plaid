import { Context, Effect, Layer, Stream, SubscriptionRef } from "effect";

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
  readonly details: unknown;
}): BrowserRuntimeError {
  return permanentClientError({
    cause: options.cause,
    message: `${options.message} for loaded index "${options.indexName}"`,
    operation: options.operation,
    details: {
      indexName: options.indexName,
      queryIndex: options.queryIndex,
      payload: options.details,
    },
  });
}

function queryEmbeddingDimension(
  payload: QueryEmbeddingsPayload,
  options: {
    readonly operation: string;
    readonly indexName: string;
    readonly queryIndex: number;
  },
): Effect.Effect<number, BrowserRuntimeError> {
  if (payload.embeddings !== undefined && payload.embeddings !== null) {
    if (payload.embeddings_b64 !== undefined && payload.embeddings_b64 !== null) {
      return Effect.fail(
        malformedQueryPayloadError({
          operation: options.operation,
          indexName: options.indexName,
          queryIndex: options.queryIndex,
          cause: "ambiguous_query_embeddings",
          message: "query payload includes both inline and binary embeddings",
          details: payload,
        }),
      );
    }

    for (const row of payload.embeddings) {
      if (row.length !== payload.encoder.embedding_dim) {
        return Effect.fail(
          malformedQueryPayloadError({
            operation: options.operation,
            indexName: options.indexName,
            queryIndex: options.queryIndex,
            cause: "query_embedding_dim_mismatch",
            message: "inline query embedding row width does not match encoder identity",
            details: payload,
          }),
        );
      }
    }

    return Effect.succeed(payload.encoder.embedding_dim);
  }

  if (payload.embeddings_b64 !== undefined && payload.embeddings_b64 !== null) {
    if (payload.shape === undefined || payload.shape === null) {
      return Effect.fail(
        malformedQueryPayloadError({
          operation: options.operation,
          indexName: options.indexName,
          queryIndex: options.queryIndex,
          cause: "missing_binary_shape",
          message: "binary query embeddings are missing shape metadata",
          details: payload,
        }),
      );
    }

    const [, dimension] = payload.shape;
    if (dimension !== payload.encoder.embedding_dim) {
      return Effect.fail(
        malformedQueryPayloadError({
          operation: options.operation,
          indexName: options.indexName,
          queryIndex: options.queryIndex,
          cause: "query_embedding_dim_mismatch",
          message: "binary query embedding shape does not match encoder identity",
          details: payload,
        }),
      );
    }

    return Effect.succeed(dimension);
  }

  return Effect.fail(
    malformedQueryPayloadError({
      operation: options.operation,
      indexName: options.indexName,
      queryIndex: options.queryIndex,
      cause: "missing_query_embeddings",
      message: "query payload does not contain embeddings",
      details: payload,
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
    const payloadDimension = yield* queryEmbeddingDimension(payload, {
      operation,
      indexName: indexMetadata.name,
      queryIndex,
    });

    if (payloadDimension !== indexMetadata.summary.dimension) {
      return yield* queryDimensionMismatchError({
        operation,
        indexName: indexMetadata.name,
        queryIndex,
        expectedDimension: indexMetadata.summary.dimension,
        actualDimension: payloadDimension,
      });
    }

    if (indexMetadata.encoder !== null && !sameEncoderIdentity(payload.encoder, indexMetadata.encoder)) {
      return yield* incompatibleEncoderError({
        operation,
        indexName: indexMetadata.name,
        expected: indexMetadata.encoder,
        actual: payload.encoder,
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
        const encoderSnapshot = yield* SubscriptionRef.get(encoderClient.state);
        if (encoderSnapshot.status === "ready") {
          yield* ensureEncoderCompatibleWithIndex(
            indexMetadata,
            encoderIdentityFromCapabilities(encoderSnapshot.capabilities),
            "browser_runtime.encode_and_search",
          );
        }

        const encoded = yield* encoderClient.encode(
          args.requestId === undefined
            ? { text: args.text }
            : { text: args.text, requestId: args.requestId },
        );

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
