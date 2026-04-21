import { expect, it } from "@effect/vitest";
import { Effect } from "effect";

import type { OnnxConfig } from "./types.js";
import { FixtureTokenizer } from "./fixture-tokenizer.js";

const tokenizerDocument = {
  kind: "whitespace_vocab",
  version: 1 as const,
  unknown_token_id: 9,
  vocab: {
    alpha: 1,
    beta: 2,
    gamma: 3,
  },
};

const defaultConfig: OnnxConfig = {
  query_prefix: "",
  document_prefix: "",
  query_length: 4,
  document_length: 4,
  do_query_expansion: false,
  embedding_dim: 4,
  uses_token_type_ids: false,
  mask_token_id: 0,
  pad_token_id: 0,
  skiplist_words: [],
  do_lower_case: false,
};

it.effect("lowercases, filters skiplist words, and emits token type ids when requested", () =>
  Effect.gen(function*() {
    const tokenizer = yield* FixtureTokenizer.fromJson(tokenizerDocument);
    const tokenized = tokenizer.encodeQuery("ALPHA beta gamma", {
      ...defaultConfig,
      do_lower_case: true,
      uses_token_type_ids: true,
      skiplist_words: ["beta"],
    });

    expect(tokenized.inputIdValues).toEqual([1, 3, 0, 0]);
    expect(tokenized.attentionMaskValues).toEqual([1, 1, 0, 0]);
    expect(tokenized.tokenTypeIdValues).toEqual([0, 0, 0, 0]);
    expect(Array.from(tokenized.inputIds)).toEqual([1n, 3n, 0n, 0n]);
    expect(Array.from(tokenized.attentionMask)).toEqual([1n, 1n, 0n, 0n]);
    expect(tokenized.tokenTypeIds).not.toBeNull();
    expect(Array.from(tokenized.tokenTypeIds ?? [])).toEqual([0n, 0n, 0n, 0n]);
  }),
);

it.effect("applies query prefixes before token lookup", () =>
  Effect.gen(function*() {
    const tokenizer = yield* FixtureTokenizer.fromJson({
      ...tokenizerDocument,
      vocab: {
        ...tokenizerDocument.vocab,
        "[q]": 7,
      },
    });

    const tokenized = tokenizer.encodeQuery("alpha", {
      ...defaultConfig,
      query_prefix: "[Q] ",
      do_lower_case: true,
    });

    expect(tokenized.inputIdValues).toEqual([7, 1, 0, 0]);
    expect(tokenized.attentionMaskValues).toEqual([1, 1, 0, 0]);
    expect(tokenized.tokenTypeIds).toBeNull();
  }),
);
