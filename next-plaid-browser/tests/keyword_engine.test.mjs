import test from "node:test";
import assert from "node:assert/strict";

import { BrowserKeywordEngine, metadataToText } from "../runtime/keyword_engine.mjs";

async function createFilterDemoEngine() {
  const engine = await BrowserKeywordEngine.create();
  engine.loadIndex({
    name: "filters-demo",
    metadata: [
      { name: "Alice", category: "A", score: 95, status: null },
      { name: "Bob", category: "B", score: 87, status: "draft" },
      { name: "Alicia", category: "A", score: 72, status: "published" }
    ],
    tokenizer: "unicode61"
  });
  return engine;
}

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

test("keyword engine resolves metadata filters like native subset selection", async () => {
  const engine = await createFilterDemoEngine();

  try {
    assert.deepEqual(
      engine.filterIndex({
        name: "filters-demo",
        condition: "category = ? AND score > ?",
        parameters: ["A", 90]
      }),
      [0],
    );

    assert.deepEqual(
      engine.filterIndex({
        name: "filters-demo",
        condition: "name REGEXP ?",
        parameters: ["^Ali"]
      }),
      [0, 2],
    );

    assert.deepEqual(
      engine.filterIndex({
        name: "filters-demo",
        condition: "status IS NULL",
        parameters: []
      }),
      [0],
    );
  } finally {
    engine.close();
  }
});

test("keyword engine accepts the native filter condition grammar", async () => {
  const engine = await createFilterDemoEngine();

  const validQueries = [
    { condition: "name = ?", parameters: ["Alice"] },
    { condition: "score > ?", parameters: [80] },
    { condition: "name = ? AND score > ?", parameters: ["Alice", 90] },
    { condition: "category = ? OR status = ?", parameters: ["A", "draft"] },
    { condition: "name LIKE ?", parameters: ["Ali%"] },
    { condition: "name NOT LIKE ?", parameters: ["Bob%"] },
    { condition: "name REGEXP ?", parameters: ["^Ali"] },
    { condition: "name NOT REGEXP ?", parameters: ["^Bob"] },
    { condition: "score BETWEEN ? AND ?", parameters: [70, 100] },
    { condition: "score NOT BETWEEN ? AND ?", parameters: [80, 90] },
    { condition: "category IN (?)", parameters: ["A"] },
    { condition: "category IN (?, ?)", parameters: ["A", "B"] },
    { condition: "category NOT IN (?, ?)", parameters: ["C", "D"] },
    { condition: "status IS NULL", parameters: [] },
    { condition: "status IS NOT NULL", parameters: [] },
    { condition: "(name = ? OR category = ?) AND score > ?", parameters: ["Bob", "A", 70] },
    { condition: "name = ? AND (category = ? OR status = ?)", parameters: ["Alice", "A", "draft"] },
    { condition: "NOT name = ?", parameters: ["Bob"] },
    { condition: "\"name\" = ?", parameters: ["Alice"] },
    { condition: "\"score\" > ?", parameters: [80] },
    { condition: "name = ? and score > ?", parameters: ["Alice", 90] },
    { condition: "score between ? and ?", parameters: [70, 100] },
    { condition: "1=1", parameters: [] },
    { condition: "1=0", parameters: [] }
  ];

  try {
    for (const query of validQueries) {
      assert.doesNotThrow(() =>
        engine.filterIndex({
          name: "filters-demo",
          condition: query.condition,
          parameters: query.parameters
        }),
      );
    }
  } finally {
    engine.close();
  }
});

test("keyword engine rejects unsafe metadata filters", async () => {
  const engine = await createFilterDemoEngine();

  try {
    assert.throws(
      () =>
        engine.filterIndex({
          name: "filters-demo",
          condition: "name = ?; DROP TABLE METADATA",
          parameters: ["Alice"]
        }),
      /Semicolons are not allowed/,
    );

    assert.throws(
      () =>
        engine.filterIndex({
          name: "filters-demo",
          condition: "name = ? -- comment",
          parameters: ["Alice"]
        }),
      /comments are not allowed/i,
    );

    assert.throws(
      () =>
        engine.filterIndex({
          name: "filters-demo",
          condition: "name = ? UNION SELECT * FROM users",
          parameters: ["Alice"]
        }),
      /UNION|SELECT/,
    );

    assert.throws(
      () =>
        engine.filterIndex({
          name: "filters-demo",
          condition: "unknown_column = ?",
          parameters: ["Alice"]
        }),
      /Unknown column/,
    );

    assert.throws(
      () =>
        engine.filterIndex({
          name: "filters-demo",
          condition: "name = 'Alice'",
          parameters: []
        }),
      /Unexpected character/,
    );

    assert.throws(
      () =>
        engine.filterIndex({
          name: "filters-demo",
          condition: "name =",
          parameters: []
        }),
      /Expected PLACEHOLDER/,
    );

    assert.throws(
      () =>
        engine.filterIndex({
          name: "filters-demo",
          condition: "LENGTH(name) > ?",
          parameters: [3]
        }),
      /Unknown column|Expected operator|Unexpected token/,
    );
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
