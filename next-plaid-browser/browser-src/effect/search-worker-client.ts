import { Context, Duration, Effect, Layer, Scope, SubscriptionRef } from "effect";
import * as Worker from "effect/unstable/workers/Worker";

import type {
  BundleInstalledResponseEnvelope,
  IndexLoadedResponseEnvelope,
  InstallBundleRequestEnvelope,
  LoadIndexRequestEnvelope,
  LoadStoredBundleRequestEnvelope,
  RuntimeErrorResponseEnvelope,
  SearchRequestEnvelope,
  SearchWorkerRequest,
  SearchResultsResponseEnvelope,
  SearchWorkerResponse,
  StorageErrorResponseEnvelope,
  StoredBundleLoadedResponseEnvelope,
} from "../shared/search-contract.js";
import {
  type SearchClientError,
  degradedClientError,
  isFatalWorkerError,
  permanentClientError,
} from "./client-errors.js";
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
    Worker.WorkerPlatform | Worker.Spawner
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
  Worker.WorkerPlatform | Worker.Spawner | Scope.Scope
> =>
  Effect.withLogSpan(
    Effect.gen(function*() {
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

      const requestResponse = Effect.fn("SearchWorkerClient.requestResponse")(
        (
          operation: string,
          requestType: string,
          _side: "runtime" | "storage",
          request:
            | LoadIndexRequestEnvelope
            | SearchRequestEnvelope
            | InstallBundleRequestEnvelope
            | LoadStoredBundleRequestEnvelope,
        ) =>
          fatalAware(
            state,
            transport.request<SearchWorkerResponse>(request, {
              operation,
              requestType,
              decodeResponse: decodePassthrough,
            }),
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

      return {
        state,
        loadIndex,
        search,
        installBundle,
        loadStoredBundle,
      } satisfies SearchWorkerClientApi;
    }),
    "search_worker.client",
  );
