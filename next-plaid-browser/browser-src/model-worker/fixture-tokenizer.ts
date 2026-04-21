import { Effect } from "effect";

import {
  WorkerRuntimeError,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import type { OnnxConfig } from "./types.js";

interface FixtureTokenizerDocument {
  kind: "whitespace_vocab";
  version: 1;
  vocab: Record<string, number>;
  unknown_token_id: number;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw workerRuntimeError({
      operation: "fixture_tokenizer.as_record",
      message: "tokenizer fixture must be an object",
      details: value,
    });
  }
  return value as Record<string, unknown>;
}

function parseFixtureTokenizer(value: unknown): FixtureTokenizerDocument {
  const record = asRecord(value);
  if (record.kind !== "whitespace_vocab" || record.version !== 1) {
    throw workerRuntimeError({
      operation: "fixture_tokenizer.parse_document",
      message: "unsupported tokenizer fixture",
      details: record,
    });
  }
  const vocabRecord = asRecord(record.vocab);
  const vocab: Record<string, number> = {};
  for (const [key, rawValue] of Object.entries(vocabRecord)) {
    if (typeof rawValue !== "number" || !Number.isInteger(rawValue)) {
      throw workerRuntimeError({
        operation: "fixture_tokenizer.parse_vocab",
        message: `tokenizer vocab value for ${key} must be an integer`,
        details: { key, rawValue },
      });
    }
    vocab[key] = rawValue;
  }
  if (typeof record.unknown_token_id !== "number" || !Number.isInteger(record.unknown_token_id)) {
    throw workerRuntimeError({
      operation: "fixture_tokenizer.parse_unknown_token",
      message: "tokenizer fixture requires an integer unknown_token_id",
      details: record.unknown_token_id,
    });
  }
  return {
    kind: "whitespace_vocab",
    version: 1,
    vocab,
    unknown_token_id: record.unknown_token_id,
  };
}

export interface TokenizedQuery {
  inputIds: BigInt64Array;
  attentionMask: BigInt64Array;
  tokenTypeIds: BigInt64Array | null;
  inputIdValues: number[];
  attentionMaskValues: number[];
  tokenTypeIdValues: number[] | null;
}

export class FixtureTokenizer {
  readonly #document: FixtureTokenizerDocument;

  private constructor(document: FixtureTokenizerDocument) {
    this.#document = document;
  }

  static fromJson(value: unknown): Effect.Effect<FixtureTokenizer, WorkerRuntimeError> {
    return Effect.try({
      try: () => new FixtureTokenizer(parseFixtureTokenizer(value)),
      catch: (error) =>
        error instanceof WorkerRuntimeError
          ? error
          : workerRuntimeErrorFromUnknown(
              "fixture_tokenizer.from_json",
              error,
              "failed to decode tokenizer fixture",
            ),
    });
  }

  static load(url: string): Effect.Effect<FixtureTokenizer, WorkerRuntimeError> {
    return Effect.tryPromise({
      try: async () => {
        const response = await fetch(url);
        if (!response.ok) {
          throw workerRuntimeError({
            operation: "fixture_tokenizer.load.fetch",
            message: `failed to fetch tokenizer fixture: ${response.status} ${response.statusText}`,
            details: { url },
          });
        }

        return response.json();
      },
      catch: (error) =>
        error instanceof WorkerRuntimeError
          ? error
          : workerRuntimeErrorFromUnknown("fixture_tokenizer.load", error),
    }).pipe(Effect.flatMap((value) => FixtureTokenizer.fromJson(value)));
  }

  encodeQuery(text: string, config: OnnxConfig): TokenizedQuery {
    const prefixed = `${config.query_prefix}${text}`;
    const normalized = config.do_lower_case ? prefixed.toLowerCase() : prefixed;
    const skiplist = new Set(
      config.skiplist_words.map((word) =>
        config.do_lower_case ? word.toLowerCase() : word,
      ),
    );
    const rawTokens = normalized
      .trim()
      .split(/\s+/u)
      .filter((token) => token.length > 0 && !skiplist.has(token))
      .slice(0, config.query_length);

    const inputIdValues: number[] = Array.from({ length: config.query_length }, () => config.pad_token_id);
    const attentionMaskValues: number[] = Array.from({ length: config.query_length }, () => 0);
    const tokenTypeIdValues = config.uses_token_type_ids
      ? Array.from({ length: config.query_length }, () => 0)
      : null;

    for (let index = 0; index < rawTokens.length; index += 1) {
      const token = rawTokens[index];
      inputIdValues[index] = this.#document.vocab[token] ?? this.#document.unknown_token_id;
      attentionMaskValues[index] = 1;
    }

    return {
      inputIds: BigInt64Array.from(inputIdValues, (value) => BigInt(value)),
      attentionMask: BigInt64Array.from(attentionMaskValues, (value) => BigInt(value)),
      tokenTypeIds:
        tokenTypeIdValues === null
          ? null
          : BigInt64Array.from(tokenTypeIdValues, (value) => BigInt(value)),
      inputIdValues,
      attentionMaskValues,
      tokenTypeIdValues,
    };
  }
}
