const FTS_TABLE = "METADATA_FTS";
const FTS_CONTENT_TABLE = "METADATA_FTS_CONTENT";
const FTS_CONTENT_COLUMN = "_fts_content_";
const FTS_CONFIG_TABLE = "_FTS_SETTINGS_";
const SQLITE_PARAM_LIMIT = 900;

let sqliteReady;
let tempTableCounter = 0;

function nextTempTableName(prefix) {
  tempTableCounter += 1;
  return `_tmp_${prefix}_${tempTableCounter}`;
}

async function getSqlite3() {
  if (!sqliteReady) {
    sqliteReady = (async () => {
      const isNode =
        typeof process !== "undefined" &&
        typeof process.versions === "object" &&
        typeof process.versions?.node === "string";
      const module = isNode
        ? await import("@sqlite.org/sqlite-wasm")
        : await import("../node_modules/@sqlite.org/sqlite-wasm/dist/index.mjs");
      return module.default();
    })();
  }

  return sqliteReady;
}

function normalizeTokenizer(tokenizer = "unicode61") {
  if (tokenizer === "unicode61" || tokenizer === "trigram") {
    return tokenizer;
  }

  throw new Error(`unsupported FTS tokenizer: ${tokenizer}`);
}

function collectTextParts(value, parts) {
  if (typeof value === "string") {
    if (value.length > 0) {
      parts.push(value);
    }
    return;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    parts.push(String(value));
    return;
  }

  if (value === null || value === undefined) {
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      collectTextParts(item, parts);
    }
    return;
  }

  if (typeof value === "object") {
    for (const nestedValue of Object.values(value)) {
      collectTextParts(nestedValue, parts);
    }
  }
}

export function metadataToText(value) {
  const parts = [];
  collectTextParts(value, parts);
  return parts.join(" ");
}

function ensureTables(db, tokenizer) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS "${FTS_CONFIG_TABLE}" (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
  `);

  db.exec(`
    CREATE TABLE IF NOT EXISTS "${FTS_CONTENT_TABLE}" (
      rowid INTEGER PRIMARY KEY,
      "${FTS_CONTENT_COLUMN}" TEXT NOT NULL DEFAULT ''
    );
  `);

  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS "${FTS_TABLE}" USING fts5(
      "${FTS_CONTENT_COLUMN}",
      content='${FTS_CONTENT_TABLE}',
      content_rowid='rowid',
      tokenize='${tokenizer}'
    );
  `);

  db.exec({
    sql: `INSERT OR REPLACE INTO "${FTS_CONFIG_TABLE}"(key, value) VALUES ('tokenizer', ?)`,
    bind: [tokenizer],
  });
}

function insertMetadata(db, metadata) {
  db.exec("BEGIN");
  try {
    const statement = db.prepare(
      `INSERT INTO "${FTS_TABLE}"(rowid, "${FTS_CONTENT_COLUMN}") VALUES (?, ?)`,
    );

    try {
      for (let documentId = 0; documentId < metadata.length; documentId += 1) {
        const value = metadata[documentId];
        const text = metadataToText(value);
        statement.bind([documentId, text]).stepReset();
      }
    } finally {
      statement.finalize();
    }

    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  }
}

function execKeywordQuery(db, query, topK, subset) {
  if (!query) {
    return [];
  }

  if (subset && subset.length === 0) {
    return [];
  }

  if (!subset || subset.length <= SQLITE_PARAM_LIMIT) {
    const subsetClause = subset
      ? ` AND rowid IN (${subset.map(() => "?").join(", ")})`
      : "";
    const bind = subset ? [query, ...subset, topK] : [query, topK];

    return db.exec({
      sql: `
        SELECT rowid AS document_id, CAST(-bm25("${FTS_TABLE}") AS REAL) AS score
        FROM "${FTS_TABLE}"
        WHERE "${FTS_TABLE}" MATCH ?${subsetClause}
        ORDER BY score DESC
        LIMIT ?
      `,
      bind,
      rowMode: "object",
      returnValue: "resultRows",
    });
  }

  const tempTable = nextTempTableName("subset");
  db.exec(`CREATE TEMP TABLE "${tempTable}" (rowid INTEGER PRIMARY KEY)`);

  try {
    db.exec("BEGIN");
    try {
      const statement = db.prepare(`INSERT INTO "${tempTable}"(rowid) VALUES (?)`);
      try {
        for (const documentId of subset) {
          statement.bind([documentId]).stepReset();
        }
      } finally {
        statement.finalize();
      }
      db.exec("COMMIT");
    } catch (error) {
      db.exec("ROLLBACK");
      throw error;
    }

    return db.exec({
      sql: `
        SELECT rowid AS document_id, CAST(-bm25("${FTS_TABLE}") AS REAL) AS score
        FROM "${FTS_TABLE}"
        WHERE "${FTS_TABLE}" MATCH ?
          AND rowid IN (SELECT rowid FROM "${tempTable}")
        ORDER BY score DESC
        LIMIT ?
      `,
      bind: [query, topK],
      rowMode: "object",
      returnValue: "resultRows",
    });
  } finally {
    db.exec(`DROP TABLE IF EXISTS "${tempTable}"`);
  }
}

class KeywordIndex {
  constructor(sqlite3, metadata, tokenizer) {
    this.sqlite3 = sqlite3;
    this.metadata = metadata ?? null;
    this.tokenizer = normalizeTokenizer(tokenizer);
    this.db = new sqlite3.oo1.DB(":memory:", "c");
    ensureTables(this.db, this.tokenizer);
    if (this.metadata) {
      insertMetadata(this.db, this.metadata);
    }
  }

  searchMany(queries, topK, subset) {
    return queries.map((query) => {
      const rows = execKeywordQuery(this.db, query, topK, subset);
      return {
        document_ids: rows.map((row) => Number(row.document_id)),
        scores: rows.map((row) => Number(row.score)),
      };
    });
  }

  metadataForDocumentIds(documentIds) {
    if (!this.metadata) {
      return documentIds.map(() => null);
    }

    return documentIds.map((documentId) => this.metadata[documentId] ?? null);
  }

  close() {
    this.db.close();
  }
}

export class BrowserKeywordEngine {
  constructor(sqlite3) {
    this.sqlite3 = sqlite3;
    this.indices = new Map();
  }

  static async create() {
    return new BrowserKeywordEngine(await getSqlite3());
  }

  loadIndex({ name, metadata, tokenizer = "unicode61" }) {
    const current = this.indices.get(name);
    if (current) {
      current.close();
      this.indices.delete(name);
    }

    if (!metadata) {
      return null;
    }

    const keywordIndex = new KeywordIndex(this.sqlite3, metadata, tokenizer);
    this.indices.set(name, keywordIndex);
    return {
      name,
      tokenizer: keywordIndex.tokenizer,
      document_count: metadata.length,
    };
  }

  searchIndex({ name, queries, topK, subset }) {
    const index = this.indices.get(name);
    if (!index) {
      return null;
    }

    return index.searchMany(queries, topK, subset);
  }

  metadataForDocumentIds(name, documentIds) {
    const index = this.indices.get(name);
    if (!index) {
      return documentIds.map(() => null);
    }

    return index.metadataForDocumentIds(documentIds);
  }

  close() {
    for (const index of this.indices.values()) {
      index.close();
    }
    this.indices.clear();
  }
}
