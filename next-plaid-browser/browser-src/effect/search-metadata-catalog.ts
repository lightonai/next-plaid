import { Context, Effect, Layer, Ref, SubscriptionRef } from "effect";

import type {
  BundleInstalledResponseEnvelope,
  BundleManifest,
  EncoderIdentity,
  IndexLoadedResponseEnvelope,
  LoadIndexRequestEnvelope,
  MutableCorpusSummary,
  StoredBundleLoadedResponseEnvelope,
} from "../shared/search-contract.js";
import type { IndexSummary } from "../generated/IndexSummary.js";

export interface LoadedSearchIndexMetadata {
  readonly name: string;
  readonly source: "load_index" | "stored_bundle";
  readonly summary: IndexSummary;
  readonly encoder: EncoderIdentity | null;
  readonly indexId: string | null;
  readonly buildId: string | null;
}

export interface MutableCorpusMetadata {
  readonly corpusId: string;
  readonly summary: MutableCorpusSummary;
  readonly loaded: boolean;
}

interface InstalledBundleMetadata {
  readonly indexId: string;
  readonly buildId: string;
  readonly manifest: BundleManifest;
}

export interface SearchMetadataCatalogApi {
  readonly loadedIndices: SubscriptionRef.SubscriptionRef<
    ReadonlyMap<string, LoadedSearchIndexMetadata>
  >;
  readonly mutableCorpora: SubscriptionRef.SubscriptionRef<
    ReadonlyMap<string, MutableCorpusMetadata>
  >;
  readonly rememberLoadedIndex: (
    request: LoadIndexRequestEnvelope,
    response: IndexLoadedResponseEnvelope,
  ) => Effect.Effect<void>;
  readonly rememberInstalledBundle: (
    response: BundleInstalledResponseEnvelope,
    manifest: BundleManifest,
  ) => Effect.Effect<void>;
  readonly rememberStoredBundleLoad: (
    response: StoredBundleLoadedResponseEnvelope,
  ) => Effect.Effect<void>;
  readonly rememberMutableCorpus: (
    metadata: MutableCorpusMetadata,
  ) => Effect.Effect<void>;
}

export class SearchMetadataCatalog
  extends Context.Service<SearchMetadataCatalog, SearchMetadataCatalogApi>()(
    "next-plaid-browser/SearchMetadataCatalog",
  )
{
  static readonly layer = Layer.effect(SearchMetadataCatalog)(
    makeSearchMetadataCatalog(),
  );
}

function storeLoadedIndex(
  loadedIndices: SubscriptionRef.SubscriptionRef<
    ReadonlyMap<string, LoadedSearchIndexMetadata>
  >,
  metadata: LoadedSearchIndexMetadata,
): Effect.Effect<void> {
  return SubscriptionRef.update(loadedIndices, (current) => {
    const next = new Map(current);
    next.set(metadata.name, metadata);
    return next;
  });
}

function storeMutableCorpus(
  mutableCorpora: SubscriptionRef.SubscriptionRef<
    ReadonlyMap<string, MutableCorpusMetadata>
  >,
  metadata: MutableCorpusMetadata,
): Effect.Effect<void> {
  return SubscriptionRef.update(mutableCorpora, (current) => {
    const next = new Map(current);
    next.set(metadata.corpusId, metadata);
    return next;
  });
}

function loadIndexMetadata(
  request: LoadIndexRequestEnvelope,
  response: IndexLoadedResponseEnvelope,
): LoadedSearchIndexMetadata {
  return {
    name: response.name,
    source: "load_index",
    summary: response.summary,
    encoder: request.encoder,
    indexId: null,
    buildId: null,
  };
}

function storedBundleLoadedMetadata(
  response: StoredBundleLoadedResponseEnvelope,
  rememberedBundle: InstalledBundleMetadata | null,
): LoadedSearchIndexMetadata {
  const encoder =
    rememberedBundle?.buildId === response.build_id
      ? rememberedBundle.manifest.encoder
      : null;

  return {
    name: response.name,
    source: "stored_bundle",
    summary: response.summary,
    encoder,
    indexId: response.index_id,
    buildId: response.build_id,
  };
}

function makeSearchMetadataCatalog(): Effect.Effect<
  SearchMetadataCatalogApi,
  never
> {
  return Effect.gen(function*() {
    const loadedIndices = yield* SubscriptionRef.make<
      ReadonlyMap<string, LoadedSearchIndexMetadata>
    >(new Map<string, LoadedSearchIndexMetadata>());
    const mutableCorpora = yield* SubscriptionRef.make<
      ReadonlyMap<string, MutableCorpusMetadata>
    >(new Map<string, MutableCorpusMetadata>());
    const installedBundles = yield* Ref.make<
      ReadonlyMap<string, InstalledBundleMetadata>
    >(new Map<string, InstalledBundleMetadata>());

    const rememberLoadedIndex = Effect.fn(
      "SearchMetadataCatalog.rememberLoadedIndex",
    )(
      (
        request: LoadIndexRequestEnvelope,
        response: IndexLoadedResponseEnvelope,
      ) => storeLoadedIndex(loadedIndices, loadIndexMetadata(request, response)),
    );

    const rememberInstalledBundle = Effect.fn(
      "SearchMetadataCatalog.rememberInstalledBundle",
    )(
      (
        response: BundleInstalledResponseEnvelope,
        manifest: BundleManifest,
      ) =>
        Ref.update(installedBundles, (current) => {
          const next = new Map(current);
          next.set(response.index_id, {
            indexId: response.index_id,
            buildId: response.build_id,
            manifest,
          });
          return next;
        }),
    );

    const rememberStoredBundleLoad = Effect.fn(
      "SearchMetadataCatalog.rememberStoredBundleLoad",
    )((response: StoredBundleLoadedResponseEnvelope) =>
      Ref.get(installedBundles).pipe(
        Effect.flatMap((current) =>
          storeLoadedIndex(
            loadedIndices,
            storedBundleLoadedMetadata(
              response,
              current.get(response.index_id) ?? null,
            ),
          )
        ),
      ));

    const rememberMutableCorpus = Effect.fn(
      "SearchMetadataCatalog.rememberMutableCorpus",
    )((metadata: MutableCorpusMetadata) =>
      storeMutableCorpus(mutableCorpora, metadata));

    return SearchMetadataCatalog.of({
      loadedIndices,
      mutableCorpora,
      rememberLoadedIndex,
      rememberInstalledBundle,
      rememberStoredBundleLoad,
      rememberMutableCorpus,
    });
  });
}
