import type { OnnxConfig } from "./types.js";

interface FixtureTokenizerDocument {
  kind: "whitespace_vocab";
  version: 1;
  vocab: Record<string, number>;
  unknown_token_id: number;
}

function asRecord(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("tokenizer fixture must be an object");
  }
  return value as Record<string, unknown>;
}

function parseFixtureTokenizer(value: unknown): FixtureTokenizerDocument {
  const record = asRecord(value);
  if (record.kind !== "whitespace_vocab" || record.version !== 1) {
    throw new Error("unsupported tokenizer fixture");
  }
  const vocabRecord = asRecord(record.vocab);
  const vocab: Record<string, number> = {};
  for (const [key, rawValue] of Object.entries(vocabRecord)) {
    if (typeof rawValue !== "number" || !Number.isInteger(rawValue)) {
      throw new Error(`tokenizer vocab value for ${key} must be an integer`);
    }
    vocab[key] = rawValue;
  }
  if (typeof record.unknown_token_id !== "number" || !Number.isInteger(record.unknown_token_id)) {
    throw new Error("tokenizer fixture requires an integer unknown_token_id");
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
  inputIdValues: number[];
  attentionMaskValues: number[];
}

export class FixtureTokenizer {
  readonly #document: FixtureTokenizerDocument;

  private constructor(document: FixtureTokenizerDocument) {
    this.#document = document;
  }

  static async load(url: string): Promise<FixtureTokenizer> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`failed to fetch tokenizer fixture: ${response.status} ${response.statusText}`);
    }

    return new FixtureTokenizer(parseFixtureTokenizer(await response.json()));
  }

  encodeQuery(text: string, config: OnnxConfig): TokenizedQuery {
    const normalized = config.do_lower_case ? text.toLowerCase() : text;
    const rawTokens = normalized
      .trim()
      .split(/\s+/u)
      .filter((token) => token.length > 0)
      .slice(0, config.query_length);

    const inputIdValues: number[] = Array.from({ length: config.query_length }, () => config.pad_token_id);
    const attentionMaskValues: number[] = Array.from({ length: config.query_length }, () => 0);

    for (let index = 0; index < rawTokens.length; index += 1) {
      const token = rawTokens[index];
      inputIdValues[index] = this.#document.vocab[token] ?? this.#document.unknown_token_id;
      attentionMaskValues[index] = 1;
    }

    return {
      inputIds: BigInt64Array.from(inputIdValues, (value) => BigInt(value)),
      attentionMask: BigInt64Array.from(attentionMaskValues, (value) => BigInt(value)),
      inputIdValues,
      attentionMaskValues,
    };
  }
}
