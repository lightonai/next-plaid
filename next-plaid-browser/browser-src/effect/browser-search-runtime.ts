import {
  Context,
  Effect,
  Layer,
  PubSub,
  Ref,
  Schema,
  SchemaIssue,
  Scope,
  Stream,
  SubscriptionRef,
} from "effect";

import type { EncoderCapabilities } from "../model-worker/types.js";
import type {
  EncoderLifecycleEvent,
  EncoderStateSnapshot,
} from "./encoder-worker-client.js";
import { EncoderWorkerClient } from "./encoder-worker-client.js";
import type {
  EncoderIdentity,
  MutableCorpusDocument,
  MutableCorpusSnapshot,
  MutableCorpusSummary,
  MutableCorpusSyncSummary,
  RegisterMutableCorpusRequestEnvelope,
  RegisterMutableCorpusResponseEnvelope,
  QueryEmbeddingsPayload,
  SearchRequestEnvelope,
  SearchResultsResponseEnvelope,
  SyncMutableCorpusResponseEnvelope,
} from "../shared/search-contract.js";
import {
  decodeDenseQueryEmbeddingsPayload,
  isBinaryQueryEmbeddingsPayload,
  isInlineQueryEmbeddingsPayload,
} from "../shared/search-contract-schema.js";
import {
  type LoadedSearchIndexMetadata,
  type MutableCorpusMetadata,
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

export interface RegisterCorpusArgs {
  readonly corpusId: string;
  readonly encoder: EncoderIdentity;
  readonly ftsTokenizer?: RegisterMutableCorpusRequestEnvelope["fts_tokenizer"] | undefined;
}

export interface SyncCorpusArgs {
  readonly corpusId: string;
  readonly snapshot: MutableCorpusSnapshot;
}

export interface SearchCorpusArgs {
  readonly corpusId: string;
  readonly queryText?: string | undefined;
  readonly request: Omit<SearchRequestEnvelope["request"], "queries">;
}

export type MutableCorpusSyncEvent =
  | {
    readonly type: "sync_started";
    readonly corpusId: string;
    readonly documentCount: number;
  }
  | {
    readonly type: "sync_committed";
    readonly corpusId: string;
    readonly summary: MutableCorpusSummary;
    readonly sync: MutableCorpusSyncSummary;
  }
  | {
    readonly type: "sync_noop";
    readonly corpusId: string;
    readonly summary: MutableCorpusSummary;
    readonly sync: MutableCorpusSyncSummary;
  }
  | {
    readonly type: "sync_failed";
    readonly corpusId: string;
    readonly error: BrowserRuntimeError;
  };

export interface BrowserSearchRuntimeApi {
  readonly encoderState: SubscriptionRef.SubscriptionRef<EncoderStateSnapshot>;
  readonly searchState: SubscriptionRef.SubscriptionRef<SearchWorkerState>;
  readonly mutableCorpora: SubscriptionRef.SubscriptionRef<
    ReadonlyMap<string, MutableCorpusMetadata>
  >;
  readonly encoderEvents: Stream.Stream<EncoderLifecycleEvent, never>;
  readonly mutableSyncEvents: Stream.Stream<MutableCorpusSyncEvent, never>;
  readonly registerCorpus: (
    args: RegisterCorpusArgs,
  ) => Effect.Effect<RegisterMutableCorpusResponseEnvelope, BrowserRuntimeError>;
  readonly syncCorpus: (
    args: SyncCorpusArgs,
  ) => Effect.Effect<SyncMutableCorpusResponseEnvelope, BrowserRuntimeError>;
  readonly searchCorpus: (
    args: SearchCorpusArgs,
  ) => Effect.Effect<SearchResultsResponseEnvelope, BrowserRuntimeError>;
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

function syncInProgressError(
  corpusId: string,
  operation: string,
): BrowserRuntimeError {
  return permanentClientError({
    cause: "sync_in_progress",
    message: `mutable corpus "${corpusId}" already has a sync in progress`,
    operation,
    details: { corpusId },
  });
}

function missingMutableCorpusTrackingError(
  corpusId: string,
  operation: string,
): BrowserRuntimeError {
  return permanentClientError({
    cause: "mutable_corpus_tracking_missing",
    message: `mutable corpus "${corpusId}" was loaded but is missing from wrapper state`,
    operation,
    details: { corpusId },
  });
}

function mutableCorpusDenseStateMissingError(
  corpusId: string,
  operation: string,
): BrowserRuntimeError {
  return permanentClientError({
    cause: "mutable_corpus_dense_state_missing",
    message: `mutable corpus "${corpusId}" does not currently have dense state`,
    operation,
    details: { corpusId },
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

function ensureEncoderCompatibleWithMutableCorpus(
  metadata: MutableCorpusMetadata,
  encoder: EncoderIdentity,
  operation: string,
): Effect.Effect<void, BrowserRuntimeError> {
  if (!sameEncoderIdentity(metadata.summary.encoder, encoder)) {
    return Effect.fail(
      incompatibleEncoderError({
        operation,
        indexName: metadata.corpusId,
        expected: metadata.summary.encoder,
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

function hasMutableDocumentEmbeddings(
  document: MutableCorpusDocument,
): boolean {
  return document.semantic_embeddings !== undefined &&
    document.semantic_embeddings !== null;
}

function mutableCorpusHasDenseState(
  summary: MutableCorpusSummary,
): boolean {
  return summary.has_dense_state;
}

function mutableCorpusSearchRequest(
  args: SearchCorpusArgs,
  queries: QueryEmbeddingsPayload[] | null,
): SearchRequestEnvelope {
  return {
    type: "search",
    name: args.corpusId,
    request: {
      ...args.request,
      queries,
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
  SearchWorkerClient | EncoderWorkerClient | Scope.Scope
> = Effect.gen(function*() {
  const searchClient = yield* SearchWorkerClient;
  const encoderClient = yield* EncoderWorkerClient;
  const mutableSyncEventPubSub = yield* Effect.acquireRelease(
    PubSub.unbounded<MutableCorpusSyncEvent>(),
    PubSub.shutdown,
  );
  const inFlightMutableCorpusSyncs = yield* Ref.make<ReadonlySet<string>>(new Set());

  const publishMutableSyncEvent = (
    event: MutableCorpusSyncEvent,
  ): Effect.Effect<void> =>
    PubSub.publish(mutableSyncEventPubSub, event).pipe(Effect.asVoid);

  const ensureMutableCorpusLoaded = Effect.fn(
    "BrowserSearchRuntime.ensureMutableCorpusLoaded",
  )((corpusId: string, operation: string) =>
    SubscriptionRef.get(searchClient.mutableCorpora).pipe(
      Effect.flatMap((current) => {
        const metadata = current.get(corpusId);
        if (metadata?.loaded) {
          return Effect.succeed(metadata);
        }

        return searchClient.loadMutableCorpus({
          type: "load_mutable_corpus",
          corpus_id: corpusId,
        }).pipe(
          Effect.andThen(SubscriptionRef.get(searchClient.mutableCorpora)),
          Effect.flatMap((next) => {
            const loaded = next.get(corpusId);
            return loaded === undefined
              ? Effect.fail(missingMutableCorpusTrackingError(corpusId, operation))
              : Effect.succeed(loaded);
          }),
        );
      }),
    ));

  const resolveMutableCorpusMetadata = Effect.fn(
    "BrowserSearchRuntime.resolveMutableCorpusMetadata",
  )((corpusId: string, operation: string) =>
    SubscriptionRef.get(searchClient.mutableCorpora).pipe(
      Effect.flatMap((current) => {
        const metadata = current.get(corpusId);
        if (metadata !== undefined) {
          return Effect.succeed(metadata);
        }

        return searchClient.loadMutableCorpus({
          type: "load_mutable_corpus",
          corpus_id: corpusId,
        }).pipe(
          Effect.andThen(SubscriptionRef.get(searchClient.mutableCorpora)),
          Effect.flatMap((next) => {
            const loaded = next.get(corpusId);
            return loaded === undefined
              ? Effect.fail(missingMutableCorpusTrackingError(corpusId, operation))
              : Effect.succeed(loaded);
          }),
        );
      }),
    ));

  const ensureMutableCorpusEncoderReady = Effect.fn(
    "BrowserSearchRuntime.ensureMutableCorpusEncoderReady",
  )((metadata: MutableCorpusMetadata, operation: string) =>
    Effect.gen(function*() {
      const snapshot = yield* SubscriptionRef.get(encoderClient.state);
      if (snapshot.status !== "ready") {
        return yield* permanentClientError({
          cause: "encoder_not_initialized",
          message: `encoder is not ready for mutable corpus "${metadata.corpusId}"`,
          operation,
          details: {
            corpusId: metadata.corpusId,
            state: snapshot.status,
          },
        });
      }

      const encoder = encoderIdentityFromCapabilities(snapshot.capabilities);
      yield* ensureEncoderCompatibleWithMutableCorpus(
        metadata,
        encoder,
        operation,
      );
      return encoder;
    }));

  const enrichMutableCorpusSnapshot = Effect.fn(
    "BrowserSearchRuntime.enrichMutableCorpusSnapshot",
  )((metadata: MutableCorpusMetadata, snapshot: MutableCorpusSnapshot, operation: string) =>
    Effect.gen(function*() {
      const needsEncoding = snapshot.documents.some((document) =>
        !hasMutableDocumentEmbeddings(document)
      );
      if (!needsEncoding) {
        return snapshot;
      }

      yield* ensureMutableCorpusEncoderReady(metadata, operation);

      const documents = yield* Effect.forEach(
        snapshot.documents,
        (document) =>
          hasMutableDocumentEmbeddings(document)
            ? Effect.succeed(document)
            : encoderClient.encodeDocument({ text: document.semantic_text }).pipe(
              Effect.map((encoded) => ({
                ...document,
                semantic_embeddings: encoded.payload,
              })),
            ),
        {
          concurrency: 1,
        },
      );

      return { documents };
    }));

  const registerCorpus = Effect.fn("BrowserSearchRuntime.registerCorpus")(
    (args: RegisterCorpusArgs) =>
      searchClient.registerMutableCorpus({
        type: "register_mutable_corpus",
        corpus_id: args.corpusId,
        encoder: args.encoder,
        fts_tokenizer: args.ftsTokenizer ?? "unicode61",
      }).pipe(
        Effect.withLogSpan("browser_runtime.register_corpus"),
        Effect.annotateLogs({
          operation: "browser_runtime.register_corpus",
          corpus_id: args.corpusId,
        }),
      ),
  );

  const syncCorpus = Effect.fn("BrowserSearchRuntime.syncCorpus")(
    (args: SyncCorpusArgs) =>
      Effect.acquireUseRelease(
        Ref.modify(inFlightMutableCorpusSyncs, (current) => {
          if (current.has(args.corpusId)) {
            return [false, current] as const;
          }
          const next = new Set(current);
          next.add(args.corpusId);
          return [true, next] as const;
        }),
        (acquired) =>
          acquired
            ? Effect.gen(function*() {
              yield* publishMutableSyncEvent({
                type: "sync_started",
                corpusId: args.corpusId,
                documentCount: args.snapshot.documents.length,
              });

              const metadata = yield* resolveMutableCorpusMetadata(
                args.corpusId,
                "browser_runtime.sync_corpus",
              );
              const snapshot = yield* enrichMutableCorpusSnapshot(
                metadata,
                args.snapshot,
                "browser_runtime.sync_corpus",
              );

              const result = yield* Effect.result(
                searchClient.syncMutableCorpus({
                  type: "sync_mutable_corpus",
                  corpus_id: args.corpusId,
                  snapshot,
                }),
              );
              if (result._tag === "Failure") {
                yield* publishMutableSyncEvent({
                  type: "sync_failed",
                  corpusId: args.corpusId,
                  error: result.failure,
                });
                return yield* result.failure;
              }

              yield* publishMutableSyncEvent({
                type: result.success.sync.changed ? "sync_committed" : "sync_noop",
                corpusId: args.corpusId,
                summary: result.success.summary,
                sync: result.success.sync,
              });
              return result.success;
            }).pipe(
              Effect.withLogSpan("browser_runtime.sync_corpus"),
              Effect.annotateLogs({
                operation: "browser_runtime.sync_corpus",
                corpus_id: args.corpusId,
                document_count: args.snapshot.documents.length,
              }),
            )
            : (() => {
              const error = syncInProgressError(
                args.corpusId,
                "browser_runtime.sync_corpus",
              );
              return publishMutableSyncEvent({
                type: "sync_failed",
                corpusId: args.corpusId,
                error,
              }).pipe(Effect.andThen(Effect.fail(error)));
            })(),
        (acquired) =>
          acquired
            ? Ref.update(inFlightMutableCorpusSyncs, (current) => {
              const next = new Set(current);
              next.delete(args.corpusId);
              return next;
            })
            : Effect.void,
      ),
  );

  const searchCorpus = Effect.fn("BrowserSearchRuntime.searchCorpus")(
    (args: SearchCorpusArgs) =>
      ensureMutableCorpusLoaded(args.corpusId, "browser_runtime.search_corpus").pipe(
        Effect.flatMap((metadata) =>
          Effect.gen(function*() {
            let queries: QueryEmbeddingsPayload[] | null = null;
            if (args.queryText !== undefined) {
              if (!mutableCorpusHasDenseState(metadata.summary)) {
                return yield* mutableCorpusDenseStateMissingError(
                  args.corpusId,
                  "browser_runtime.search_corpus",
                );
              }

              yield* ensureMutableCorpusEncoderReady(
                metadata,
                "browser_runtime.search_corpus",
              );
              const encoded = yield* encoderClient.encodeQuery({
                text: args.queryText,
              });
              queries = [encoded.payload];
            }

            return yield* searchClient.search(
              mutableCorpusSearchRequest(args, queries),
            );
          })
        ),
        Effect.withLogSpan("browser_runtime.search_corpus"),
        Effect.annotateLogs({
          operation: "browser_runtime.search_corpus",
          corpus_id: args.corpusId,
          top_k: args.request.params.top_k,
          has_dense_query_text: args.queryText === undefined ? 0 : args.queryText.length,
          has_text_query: args.request.text_query?.length ?? 0,
        }),
      ),
  );

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
          encoderClient.encodeQuery(
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
    mutableCorpora: searchClient.mutableCorpora,
    encoderEvents: encoderClient.events,
    mutableSyncEvents: Stream.fromPubSub(mutableSyncEventPubSub),
    registerCorpus,
    syncCorpus,
    searchCorpus,
    searchWithEmbeddings,
    encodeAndSearch,
  } satisfies BrowserSearchRuntimeApi;
});
