// TypeScript-owned model-worker scaffolding. These types do not cross into the
// current Rust/Wasm runtime boundary; shared search/storage payloads must come
// from `../shared/search-contract.ts` so they stay derived from Rust.
import type { EncoderIdentity, QueryEmbeddingsPayload } from "../shared/search-contract.js";

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

export interface EncoderCreateInput {
  encoder: EncoderIdentity;
  modelUrl: string;
  onnxConfigUrl: string;
  tokenizerUrl: string;
  prefer?: "wasm" | "auto";
}

export interface EncoderBackend {
  readonly capabilities: EncoderCapabilities;
  encode(text: string): Promise<EncodedQuery>;
  health(): Promise<EncoderHealth>;
  dispose(): Promise<void>;
}

export type EncoderInitEvent =
  | { stage: "fetch_start"; url: string; expectedBytes: number | null }
  | { stage: "fetch_complete"; url: string; bytesReceived: number }
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

export interface EncoderDisposeResponse {
  type: "encoder_disposed";
}

export interface EncoderInitRequest {
  type: "init";
  payload: EncoderCreateInput;
}

export interface EncoderEncodeRequest {
  type: "encode";
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
  | EncoderEncodeRequest
  | EncoderHealthRequest
  | EncoderDisposeRequest;

export type EncoderWorkerResponse =
  | EncoderInitResponse
  | EncoderHealthResponse
  | EncodeResponse
  | EncoderDisposeResponse;
