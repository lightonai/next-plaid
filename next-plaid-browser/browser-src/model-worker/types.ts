// TypeScript-owned model-worker scaffolding. These types do not cross into the
// current Rust/Wasm runtime boundary; shared search/storage payloads must come
// from `../shared/search-contract.ts` so they stay derived from Rust.
import type * as Effect from "effect/Effect";

import type { WorkerRuntimeError } from "../effect/worker-runtime-errors.js";
import type { MatrixPayload } from "../generated/MatrixPayload.js";
import type { EncoderIdentity, QueryEmbeddingsPayload } from "../shared/search-contract.js";
import type { DurableModelAssetStoreKind, ModelAssetStoreKind } from "./model-asset-types.js";

export type BackendKind = "wasm";
export type EncoderState = "empty" | "initializing" | "ready" | "failed" | "disposed";
export type EncoderHealth = "ok" | "degraded";

export interface OnnxConfig {
  query_prefix: string;
  document_prefix: string;
  query_length: number;
  document_length: number;
  do_query_expansion: boolean;
  embedding_dim: number;
  uses_token_type_ids: boolean;
  mask_token_id: number;
  pad_token_id: number;
  query_prefix_id?: number;
  document_prefix_id?: number;
  skiplist_words: string[];
  do_lower_case: boolean;
}

export interface EncoderCapabilities {
  backend: BackendKind;
  threaded: boolean;
  persistentStorage: boolean;
  encoderId: string;
  encoderBuild: string;
  embeddingDim: number;
  queryLength: number;
  doQueryExpansion: boolean;
  normalized: boolean;
}

export interface EncodeTimingBreakdown {
  total_ms: number;
  tokenize_ms: number;
  inference_ms: number;
}

export interface EncodedQuery {
  payload: QueryEmbeddingsPayload;
  timing: EncodeTimingBreakdown;
  input_ids: number[];
  attention_mask: number[];
}

export interface EncodedDocument {
  payload: MatrixPayload;
  timing: EncodeTimingBreakdown;
  input_ids: number[];
  attention_mask: number[];
}

export interface EncoderCreateInput {
  encoder: EncoderIdentity;
  modelUrl: string;
  onnxConfigUrl: string;
  tokenizerUrl: string;
  prefer?: "wasm" | "auto";
}

export interface EncoderBackend {
  readonly capabilities: EncoderCapabilities;
  encodeQuery(text: string): Effect.Effect<EncodedQuery, WorkerRuntimeError>;
  encodeDocument(text: string): Effect.Effect<EncodedDocument, WorkerRuntimeError>;
  health(): Effect.Effect<EncoderHealth, WorkerRuntimeError>;
}

export type EncoderInitEvent =
  | { stage: "asset_memory_hit"; url: string; bytesReceived: number }
  | {
      stage: "asset_store_hit";
      url: string;
      storeKind: ModelAssetStoreKind;
      bytesReceived: number;
    }
  | {
      stage: "asset_store_miss";
      url: string;
      storeKind: ModelAssetStoreKind;
    }
  | { stage: "asset_fetch_start"; url: string; expectedBytes: number | null }
  | { stage: "asset_fetch_complete"; url: string; bytesReceived: number }
  | {
      stage: "asset_store_write_start";
      url: string;
      storeKind: DurableModelAssetStoreKind;
      bytesReceived: number;
    }
  | {
      stage: "asset_store_write_complete";
      url: string;
      storeKind: DurableModelAssetStoreKind;
      bytesReceived: number;
    }
  | { stage: "config_validated"; queryLength: number; embeddingDim: number }
  | { stage: "session_create_start" }
  | { stage: "session_create_complete"; durationMs: number }
  | { stage: "warmup_start" }
  | { stage: "warmup_complete"; durationMs: number }
  | { stage: "ready"; capabilities: EncoderCapabilities };

export interface EncoderInitResponse {
  type: "encoder_ready";
  state: "ready";
  capabilities: EncoderCapabilities;
}

export interface EncoderHealthResponse {
  type: "encoder_health";
  state: EncoderState;
  health: EncoderHealth;
  capabilities: EncoderCapabilities | null;
  last_error: string | null;
}

export interface EncodeResponse {
  type: "encoded_query";
  encoded: EncodedQuery;
}

export interface EncodeDocumentResponse {
  type: "encoded_document";
  encoded: EncodedDocument;
}

export interface EncoderDisposeResponse {
  type: "encoder_disposed";
}

export interface EncoderInitRequest {
  type: "init";
  payload: EncoderCreateInput;
}

export interface EncoderEncodeQueryRequest {
  type: "encode_query";
  payload: { text: string };
}

export interface EncoderEncodeDocumentRequest {
  type: "encode_document";
  payload: { text: string };
}

export interface EncoderHealthRequest {
  type: "health";
}

export interface EncoderDisposeRequest {
  type: "dispose";
}

export type EncoderWorkerRequest =
  | EncoderInitRequest
  | EncoderEncodeQueryRequest
  | EncoderEncodeDocumentRequest
  | EncoderHealthRequest
  | EncoderDisposeRequest;

export type EncoderWorkerResponse =
  | EncoderInitResponse
  | EncoderHealthResponse
  | EncodeResponse
  | EncodeDocumentResponse
  | EncoderDisposeResponse;
