// Shared Rust/Wasm contract surface. The generated files come from
// `next-plaid-browser-contract` via `cargo run --bin export_bindings`.
// This wrapper exists only for local re-exports and convenience aliases.
export type { BundleInstalledResponse } from "../generated/BundleInstalledResponse";
export type { BundleManifest } from "../generated/BundleManifest";
export type { EncoderIdentity } from "../generated/EncoderIdentity";
export type { HealthResponse } from "../generated/HealthResponse";
export type { InstallBundleRequest } from "../generated/InstallBundleRequest";
export type { LoadMutableCorpusRequest } from "../generated/LoadMutableCorpusRequest";
export type { LoadMutableCorpusResponse } from "../generated/LoadMutableCorpusResponse";
export type { LoadStoredBundleRequest } from "../generated/LoadStoredBundleRequest";
export type { MutableCorpusDocument } from "../generated/MutableCorpusDocument";
export type { MutableCorpusSnapshot } from "../generated/MutableCorpusSnapshot";
export type { MutableCorpusSummary } from "../generated/MutableCorpusSummary";
export type { MutableCorpusSyncSummary } from "../generated/MutableCorpusSyncSummary";
export type { QueryEmbeddingsPayload } from "../generated/QueryEmbeddingsPayload";
export type { RegisterMutableCorpusRequest } from "../generated/RegisterMutableCorpusRequest";
export type { RegisterMutableCorpusResponse } from "../generated/RegisterMutableCorpusResponse";
export type { RuntimeErrorResponse } from "../generated/RuntimeErrorResponse";
export type { RuntimeRequest } from "../generated/RuntimeRequest";
export type { RuntimeResponse } from "../generated/RuntimeResponse";
export type { SearchResponse } from "../generated/SearchResponse";
export type { StorageRequest } from "../generated/StorageRequest";
export type { StorageResponse } from "../generated/StorageResponse";
export type { StoredBundleLoadedResponse } from "../generated/StoredBundleLoadedResponse";
export type { SyncMutableCorpusRequest } from "../generated/SyncMutableCorpusRequest";
export type { SyncMutableCorpusResponse } from "../generated/SyncMutableCorpusResponse";
export type { WorkerLoadIndexRequest } from "../generated/WorkerLoadIndexRequest";
export type { WorkerLoadIndexResponse } from "../generated/WorkerLoadIndexResponse";
export type { WorkerSearchRequest } from "../generated/WorkerSearchRequest";

import type { RuntimeRequest } from "../generated/RuntimeRequest";
import type { RuntimeResponse } from "../generated/RuntimeResponse";
import type { StorageRequest } from "../generated/StorageRequest";
import type { StorageResponse } from "../generated/StorageResponse";

export type SearchWorkerRequest = RuntimeRequest | StorageRequest;
export type SearchWorkerResponse = RuntimeResponse | StorageResponse;

export type HealthRequestEnvelope = Extract<RuntimeRequest, { type: "health" }>;
export type LoadIndexRequestEnvelope = Extract<RuntimeRequest, { type: "load_index" }>;
export type SearchRequestEnvelope = Extract<RuntimeRequest, { type: "search" }>;
export type InstallBundleRequestEnvelope = Extract<StorageRequest, { type: "install_bundle" }>;
export type RegisterMutableCorpusRequestEnvelope = Extract<
  StorageRequest,
  { type: "register_mutable_corpus" }
>;
export type SyncMutableCorpusRequestEnvelope = Extract<
  StorageRequest,
  { type: "sync_mutable_corpus" }
>;
export type LoadMutableCorpusRequestEnvelope = Extract<
  StorageRequest,
  { type: "load_mutable_corpus" }
>;
export type LoadStoredBundleRequestEnvelope = Extract<StorageRequest, { type: "load_stored_bundle" }>;

export type HealthResponseEnvelope = Extract<RuntimeResponse, { type: "health" }>;
export type RuntimeErrorResponseEnvelope = Extract<RuntimeResponse, { type: "error" }>;
export type StorageErrorResponseEnvelope = Extract<StorageResponse, { type: "error" }>;
export type IndexLoadedResponseEnvelope = Extract<RuntimeResponse, { type: "index_loaded" }>;
export type SearchResultsResponseEnvelope = Extract<RuntimeResponse, { type: "search_results" }>;
export type BundleInstalledResponseEnvelope = Extract<StorageResponse, { type: "bundle_installed" }>;
export type RegisterMutableCorpusResponseEnvelope = Extract<
  StorageResponse,
  { type: "mutable_corpus_registered" }
>;
export type SyncMutableCorpusResponseEnvelope = Extract<
  StorageResponse,
  { type: "mutable_corpus_synced" }
>;
export type LoadMutableCorpusResponseEnvelope = Extract<
  StorageResponse,
  { type: "mutable_corpus_loaded" }
>;
export type StoredBundleLoadedResponseEnvelope = Extract<
  StorageResponse,
  { type: "stored_bundle_loaded" }
>;
