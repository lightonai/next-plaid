import type { OnnxConfig } from "./types.js";

const DEFAULTS: OnnxConfig = {
  query_prefix: "[Q] ",
  document_prefix: "[D] ",
  query_length: 48,
  document_length: 300,
  do_query_expansion: true,
  embedding_dim: 128,
  uses_token_type_ids: true,
  mask_token_id: 103,
  pad_token_id: 0,
  skiplist_words: [],
  do_lower_case: false,
};

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("onnx_config.json must be an object");
  }
  return value as Record<string, unknown>;
}

function readString(record: Record<string, unknown>, key: string, fallback: string): string {
  const value = record[key];
  return typeof value === "string" ? value : fallback;
}

function readNumber(record: Record<string, unknown>, key: string, fallback: number): number {
  const value = record[key];
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return fallback;
}

function readBoolean(record: Record<string, unknown>, key: string, fallback: boolean): boolean {
  const value = record[key];
  return typeof value === "boolean" ? value : fallback;
}

export function parseOnnxConfig(value: unknown): OnnxConfig {
  const record = asRecord(value);
  const skiplistValue = record.skiplist_words;
  const skiplist_words = Array.isArray(skiplistValue)
    ? skiplistValue.filter((item): item is string => typeof item === "string")
    : DEFAULTS.skiplist_words;

  return {
    query_prefix: readString(record, "query_prefix", DEFAULTS.query_prefix),
    document_prefix: readString(record, "document_prefix", DEFAULTS.document_prefix),
    query_length: readNumber(record, "query_length", DEFAULTS.query_length),
    document_length: readNumber(record, "document_length", DEFAULTS.document_length),
    do_query_expansion: readBoolean(record, "do_query_expansion", DEFAULTS.do_query_expansion),
    embedding_dim: readNumber(record, "embedding_dim", DEFAULTS.embedding_dim),
    uses_token_type_ids: readBoolean(
      record,
      "uses_token_type_ids",
      DEFAULTS.uses_token_type_ids,
    ),
    mask_token_id: readNumber(record, "mask_token_id", DEFAULTS.mask_token_id),
    pad_token_id: readNumber(record, "pad_token_id", DEFAULTS.pad_token_id),
    skiplist_words,
    do_lower_case: readBoolean(record, "do_lower_case", DEFAULTS.do_lower_case),
  };
}

export async function loadOnnxConfig(url: string): Promise<OnnxConfig> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch onnx config: ${response.status} ${response.statusText}`);
  }

  return parseOnnxConfig(await response.json());
}
