import test from "node:test";
import assert from "node:assert/strict";

import { BrowserKeywordEngine, metadataToText } from "../runtime/keyword_engine.mjs";

test("metadataToText flattens nested values like native metadata indexing", () => {
  const text = metadataToText({
    title: "Doc",
    count: 42,
    active: true,
    nested: {
      tags: ["rust", "search"],
      note: null
    }
  });

  assert.equal(text, "Doc 42 true rust search");
});

test("keyword engine returns BM25-ranked hits", async () => {
  const engine = await BrowserKeywordEngine.create();

  try {
    engine.loadIndex({
      name: "demo",
      metadata: [
        { title: "alpha launch memo", topic: "edge" },
        { title: "beta report summary", topic: "metrics" },
        { title: "alpha beta digest", topic: "mixed" }
      ],
      tokenizer: "unicode61"
    });

    const results = engine.searchIndex({
      name: "demo",
      queries: ["alpha"],
      topK: 3,
      subset: null
    });

    assert.deepEqual(results[0].document_ids, [0, 2]);
    assert.equal(results[0].scores.length, 2);
    assert.ok(results[0].scores.every((score) => score > 0));
  } finally {
    engine.close();
  }
});

test("keyword engine respects subset filtering", async () => {
  const engine = await BrowserKeywordEngine.create();

  try {
    engine.loadIndex({
      name: "subset-demo",
      metadata: [
        { title: "alpha launch memo" },
        { title: "beta report summary" },
        { title: "alpha beta digest" }
      ],
      tokenizer: "unicode61"
    });

    const results = engine.searchIndex({
      name: "subset-demo",
      queries: ["alpha"],
      topK: 5,
      subset: [2]
    });

    assert.deepEqual(results[0].document_ids, [2]);
  } finally {
    engine.close();
  }
});

test("keyword engine supports trigram tokenizer substring search", async () => {
  const engine = await BrowserKeywordEngine.create();

  try {
    engine.loadIndex({
      name: "trigram-demo",
      metadata: [
        { symbol: "parse_arguments" },
        { symbol: "render_template" },
        { symbol: "validate_input" }
      ],
      tokenizer: "trigram"
    });

    const results = engine.searchIndex({
      name: "trigram-demo",
      queries: ["templ"],
      topK: 3,
      subset: null
    });

    assert.deepEqual(results[0].document_ids, [1]);
  } finally {
    engine.close();
  }
});
