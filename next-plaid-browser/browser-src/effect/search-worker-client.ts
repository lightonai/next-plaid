import { Context, Duration, Effect, Layer, Scope, SubscriptionRef } from "effect";
import * as Worker from "effect/unstable/workers/Worker";

import type {
  BundleInstalledResponseEnvelope,
  IndexLoadedResponseEnvelope,
  InstallBundleRequestEnvelope,
  LoadMutableCorpusRequestEnvelope,
  LoadMutableCorpusResponseEnvelope,
  LoadIndexRequestEnvelope,
  LoadStoredBundleRequestEnvelope,
  RegisterMutableCorpusRequestEnvelope,
  RegisterMutableCorpusResponseEnvelope,
  RuntimeErrorResponseEnvelope,
  SearchRequestEnvelope,
  SearchWorkerRequest,
  SearchResultsResponseEnvelope,
  SearchWorkerResponse,
  StorageErrorResponseEnvelope,
  StoredBundleLoadedResponseEnvelope,
  SyncMutableCorpusRequestEnvelope,
  SyncMutableCorpusResponseEnvelope,
} from "../shared/search-contract.js";
import {
  type SearchClientError,
  degradedClientError,
  isFatalWorkerError,
  permanentClientError,
  transientClientError,
} from "./client-errors.js";
import {
  SearchMetadataCatalog,
} from "./search-metadata-catalog.js";
import { decodePassthrough, makeWorkerTransport } from "./worker-transport.js";

export type SearchWorkerState =
  | { status: "starting"; lastError: null }
  | { status: "ready"; lastError: null }
  | { status: "failed"; lastError: SearchClientError }
  | { status: "disposed"; lastError: null };

export interface SearchWorkerClientApi {
  readonly state: SubscriptionRef.SubscriptionRef<SearchWorkerState>;
  readonly loadIndex: (
    request: LoadIndexRequestEnvelope,
  ) => Effect.Effect<IndexLoadedResponseEnvelope, SearchClientError>;
  readonly search: (
    request: SearchRequestEnvelope,
  ) => Effect.Effect<SearchResultsResponseEnvelope, SearchClientError>;
  readonly installBundle: (
    request: InstallBundleRequestEnvelope,
  ) => Effect.Effect<BundleInstalledResponseEnvelope, SearchClientError>;
  readonly loadStoredBundle: (
    request: LoadStoredBundleRequestEnvelope,
  ) => Effect.Effect<StoredBundleLoadedResponseEnvelope, SearchClientError>;
  readonly registerMutableCorpus: (
    request: RegisterMutableCorpusRequestEnvelope,
  ) => Effect.Effect<RegisterMutableCorpusResponseEnvelope, SearchClientError>;
  readonly syncMutableCorpus: (
    request: SyncMutableCorpusRequestEnvelope,
  ) => Effect.Effect<SyncMutableCorpusResponseEnvelope, SearchClientError>;
  readonly loadMutableCorpus: (
    request: LoadMutableCorpusRequestEnvelope,
  ) => Effect.Effect<LoadMutableCorpusResponseEnvelope, SearchClientError>;
}

export class SearchWorkerClient
  extends Context.Service<SearchWorkerClient, SearchWorkerClientApi>()(
    "next-plaid-browser/SearchWorkerClient",
  )
{
  static layer = (
    options: SearchWorkerClientOptions = {},
  ): Layer.Layer<
    SearchWorkerClient,
    SearchClientError,
    SearchMetadataCatalog | Worker.WorkerPlatform | Worker.Spawner
  > => Layer.effect(SearchWorkerClient)(makeSearchWorkerClient(options));
}

export interface SearchWorkerClientOptions {
  readonly requestTimeout?: Duration.Input | undefined;
}

function mapWireError(
  side: "runtime" | "storage",
  operation: string,
  response: RuntimeErrorResponseEnvelope | StorageErrorResponseEnvelope,
): SearchClientError {
  const cause = side === "storage" ? "storage_error_response" : "runtime_error_response";
  switch (response.code) {
    case "internal":
    case "kernel_failed":
    case "keyword_runtime_failed":
    case "bundle_load_failed":
    case "storage_failed":
      return degradedClientError({
        cause,
        message: response.message,
        operation,
        details: response,
      });
    default:
      return permanentClientError({
        cause,
        message: response.message,
        operation,
        details: response,
      });
  }
}

function fatalAware<T>(
  state: SubscriptionRef.SubscriptionRef<SearchWorkerState>,
  effect: Effect.Effect<T, SearchClientError>,
): Effect.Effect<T, SearchClientError> {
  const handleError = (error: SearchClientError): Effect.Effect<T, SearchClientError> =>
    isFatalWorkerError(error)
      ? SubscriptionRef.set<SearchWorkerState>(state, {
          status: "failed",
          lastError: error,
        }).pipe(Effect.andThen(Effect.fail(error)))
      : Effect.fail(error);

  return effect.pipe(
    Effect.catchTags({
      TransientClientError: handleError,
      PermanentClientError: handleError,
      DegradedClientError: handleError,
    }),
  );
}

function expectResponseType<TResponse extends SearchWorkerResponse>(
  side: "runtime" | "storage",
  response: SearchWorkerResponse,
  expectedType: TResponse["type"],
  operation: string,
): Effect.Effect<TResponse, SearchClientError> {
  if (response.type === "error") {
    return Effect.fail(mapWireError(side, operation, response));
  }
  if (response.type !== expectedType) {
    return Effect.fail(
      permanentClientError({
        cause: "unexpected_response_type",
        message: `expected ${expectedType}, received ${response.type}`,
        operation,
        details: response,
      }),
    );
  }
  return Effect.succeed(response as TResponse);
}

export const makeSearchWorkerClient = (
  options: SearchWorkerClientOptions = {},
): Effect.Effect<
  SearchWorkerClientApi,
  SearchClientError,
  SearchMetadataCatalog | Worker.WorkerPlatform | Worker.Spawner | Scope.Scope
> =>
  Effect.withLogSpan(
    Effect.gen(function*() {
      const metadataCatalog = yield* SearchMetadataCatalog;
      const state = yield* Effect.acquireRelease(
        SubscriptionRef.make<SearchWorkerState>({ status: "starting", lastError: null }),
        (ref) => SubscriptionRef.set(ref, { status: "disposed", lastError: null }),
      );
      const transport = yield* makeWorkerTransport<SearchWorkerRequest>({
        workerKind: "search",
        requestTimeout: options.requestTimeout,
        onWorkerFailure: (error) =>
          SubscriptionRef.set<SearchWorkerState>(state, {
            status: "failed",
            lastError: error,
          }).pipe(Effect.asVoid),
      });

      yield* SubscriptionRef.set(state, { status: "ready", lastError: null });

      const ensureClientUsable = Effect.fn("SearchWorkerClient.ensureClientUsable")(
        (operation: string) =>
          Effect.gen(function*() {
            const snapshot = yield* SubscriptionRef.get(state);
            if (snapshot.status === "failed") {
              return yield* snapshot.lastError;
            }
            if (snapshot.status === "disposed") {
              return yield* transientClientError({
                cause: "worker_disposed",
                message: "search worker client scope is closed",
                operation,
                details: null,
              });
            }
          }),
      );

      const requestResponse = Effect.fn("SearchWorkerClient.requestResponse")(
        (
          operation: string,
          requestType: string,
          _side: "runtime" | "storage",
          request:
            | LoadIndexRequestEnvelope
            | SearchRequestEnvelope
            | InstallBundleRequestEnvelope
            | LoadStoredBundleRequestEnvelope
            | RegisterMutableCorpusRequestEnvelope
            | SyncMutableCorpusRequestEnvelope
            | LoadMutableCorpusRequestEnvelope,
        ) =>
          ensureClientUsable(operation).pipe(
            Effect.andThen(
              fatalAware(
                state,
                transport.request<SearchWorkerResponse>(request, {
                  operation,
                  requestType,
                  decodeResponse: decodePassthrough,
                }),
              ),
            ),
          ),
      );

      const loadIndex = Effect.fn("SearchWorkerClient.loadIndex")(
        (request: LoadIndexRequestEnvelope) =>
          requestResponse(
            "search_worker.load_index",
            request.type,
            "runtime",
            request,
          ).pipe(
            Effect.flatMap((response) =>
              expectResponseType<IndexLoadedResponseEnvelope>(
                "runtime",
                response,
                "index_loaded",
                "search_worker.load_index",
              ).pipe(
                Effect.tap((decoded) =>
                  metadataCatalog.rememberLoadedIndex(request, decoded),
                ),
              ),
            ),
            Effect.withLogSpan("search_worker.load_index"),
            Effect.annotateLogs({
              worker_kind: "search",
              operation: "search_worker.load_index",
              index_name: request.name,
            }),
          ),
      );

      const search = Effect.fn("SearchWorkerClient.search")(
        (request: SearchRequestEnvelope) =>
          requestResponse(
            "search_worker.search",
            request.type,
            "runtime",
            request,
          ).pipe(
            Effect.flatMap((response) =>
              expectResponseType<SearchResultsResponseEnvelope>(
                "runtime",
                response,
                "search_results",
                "search_worker.search",
              ),
            ),
            Effect.withLogSpan("search_worker.search"),
            Effect.annotateLogs({
              worker_kind: "search",
              operation: "search_worker.search",
              index_name: request.name,
              top_k: request.request.params.top_k,
              query_count: request.request.queries?.length ?? 0,
            }),
          ),
      );

      const installBundle = Effect.fn("SearchWorkerClient.installBundle")(
        (request: InstallBundleRequestEnvelope) =>
          requestResponse(
            "search_worker.install_bundle",
            request.type,
            "storage",
            request,
          ).pipe(
            Effect.flatMap((response) =>
              expectResponseType<BundleInstalledResponseEnvelope>(
                "storage",
                response,
                "bundle_installed",
                "search_worker.install_bundle",
              ).pipe(
                Effect.tap((decoded) =>
                  metadataCatalog.rememberInstalledBundle(
                    decoded,
                    request.manifest,
                  ),
                ),
              ),
            ),
            Effect.withLogSpan("search_worker.install_bundle"),
            Effect.annotateLogs({
              worker_kind: "search",
              operation: "search_worker.install_bundle",
              bundle_index_id: request.manifest.index_id,
            }),
          ),
      );

      const loadStoredBundle = Effect.fn("SearchWorkerClient.loadStoredBundle")(
        (request: LoadStoredBundleRequestEnvelope) =>
          requestResponse(
            "search_worker.load_stored_bundle",
            request.type,
            "storage",
            request,
          ).pipe(
            Effect.flatMap((response) =>
              expectResponseType<StoredBundleLoadedResponseEnvelope>(
                "storage",
                response,
                "stored_bundle_loaded",
                "search_worker.load_stored_bundle",
              ).pipe(
                Effect.tap((decoded) =>
                  metadataCatalog.rememberStoredBundleLoad(decoded),
                ),
              ),
            ),
            Effect.withLogSpan("search_worker.load_stored_bundle"),
            Effect.annotateLogs({
              worker_kind: "search",
              operation: "search_worker.load_stored_bundle",
              bundle_index_id: request.index_id,
              index_name: request.name,
            }),
          ),
      );

      const registerMutableCorpus = Effect.fn("SearchWorkerClient.registerMutableCorpus")(
        (request: RegisterMutableCorpusRequestEnvelope) =>
          requestResponse(
            "search_worker.register_mutable_corpus",
            request.type,
            "storage",
            request,
          ).pipe(
            Effect.flatMap((response) =>
              expectResponseType<RegisterMutableCorpusResponseEnvelope>(
                "storage",
                response,
                "mutable_corpus_registered",
                "search_worker.register_mutable_corpus",
              ).pipe(
                Effect.tap((decoded) =>
                  metadataCatalog.rememberMutableCorpus({
                    corpusId: decoded.corpus_id,
                    summary: decoded.summary,
                    loaded: false,
                  }),
                ),
              ),
            ),
            Effect.withLogSpan("search_worker.register_mutable_corpus"),
            Effect.annotateLogs({
              worker_kind: "search",
              operation: "search_worker.register_mutable_corpus",
              corpus_id: request.corpus_id,
            }),
          ),
      );

      const syncMutableCorpus = Effect.fn("SearchWorkerClient.syncMutableCorpus")(
        (request: SyncMutableCorpusRequestEnvelope) =>
          requestResponse(
            "search_worker.sync_mutable_corpus",
            request.type,
            "storage",
            request,
          ).pipe(
            Effect.flatMap((response) =>
              expectResponseType<SyncMutableCorpusResponseEnvelope>(
                "storage",
                response,
                "mutable_corpus_synced",
                "search_worker.sync_mutable_corpus",
              ).pipe(
                Effect.tap((decoded) =>
                  metadataCatalog.rememberMutableCorpus({
                    corpusId: decoded.corpus_id,
                    summary: decoded.summary,
                    loaded: true,
                  }),
                ),
              ),
            ),
            Effect.withLogSpan("search_worker.sync_mutable_corpus"),
            Effect.annotateLogs({
              worker_kind: "search",
              operation: "search_worker.sync_mutable_corpus",
              corpus_id: request.corpus_id,
              document_count: request.snapshot.documents.length,
            }),
          ),
      );

      const loadMutableCorpus = Effect.fn("SearchWorkerClient.loadMutableCorpus")(
        (request: LoadMutableCorpusRequestEnvelope) =>
          requestResponse(
            "search_worker.load_mutable_corpus",
            request.type,
            "storage",
            request,
          ).pipe(
            Effect.flatMap((response) =>
              expectResponseType<LoadMutableCorpusResponseEnvelope>(
                "storage",
                response,
                "mutable_corpus_loaded",
                "search_worker.load_mutable_corpus",
              ).pipe(
                Effect.tap((decoded) =>
                  metadataCatalog.rememberMutableCorpus({
                    corpusId: decoded.corpus_id,
                    summary: decoded.summary,
                    loaded: true,
                  }),
                ),
              ),
            ),
            Effect.withLogSpan("search_worker.load_mutable_corpus"),
            Effect.annotateLogs({
              worker_kind: "search",
              operation: "search_worker.load_mutable_corpus",
              corpus_id: request.corpus_id,
            }),
          ),
      );

      return {
        state,
        loadIndex,
        search,
        installBundle,
        loadStoredBundle,
        registerMutableCorpus,
        syncMutableCorpus,
        loadMutableCorpus,
      } satisfies SearchWorkerClientApi;
    }),
    "search_worker.client",
  );
