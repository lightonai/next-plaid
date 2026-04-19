const METADATA_TABLE = "METADATA";
const SUBSET_COLUMN = "_subset_";
const FTS_TABLE = "METADATA_FTS";
const FTS_CONTENT_TABLE = "METADATA_FTS_CONTENT";
const FTS_CONTENT_COLUMN = "_fts_content_";
const FTS_CONFIG_TABLE = "_FTS_SETTINGS_";
const SQLITE_PARAM_LIMIT = 900;
const COLUMN_NAME_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_]*$/;
const NUMERIC_EQUALITY_PATTERN = /^(\d+)\s*=\s*(\d+)$/;
const DANGEROUS_KEYWORDS = [
  "SELECT",
  "UNION",
  "INSERT",
  "UPDATE",
  "DELETE",
  "DROP",
  "CREATE",
  "ALTER",
  "TRUNCATE",
  "EXEC",
  "EXECUTE",
  "GRANT",
  "REVOKE"
];

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

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isValidColumnName(name) {
  return COLUMN_NAME_PATTERN.test(name);
}

function inferSqlType(value) {
  if (typeof value === "number") {
    return Number.isInteger(value) ? "INTEGER" : "REAL";
  }

  if (typeof value === "boolean") {
    return "INTEGER";
  }

  if (typeof value === "string" || value === null || value === undefined) {
    return "TEXT";
  }

  return "BLOB";
}

function jsonToSql(value) {
  if (value === null || value === undefined) {
    return null;
  }

  if (typeof value === "boolean") {
    return value ? 1 : 0;
  }

  if (typeof value === "number" || typeof value === "string") {
    return value;
  }

  return JSON.stringify(value);
}

function collectMetadataSchema(metadata) {
  const columns = [];
  const columnTypes = new Map();
  const seenColumns = new Set();

  for (const item of metadata) {
    if (!isPlainObject(item)) {
      continue;
    }

    for (const [columnName, value] of Object.entries(item)) {
      if (!isValidColumnName(columnName)) {
        throw new Error(
          `Invalid metadata column '${columnName}'. Column names must start with a letter or underscore and only contain letters, digits, or underscores.`,
        );
      }

      if (!seenColumns.has(columnName)) {
        seenColumns.add(columnName);
        columns.push(columnName);
      }

      if (!columnTypes.has(columnName) && value !== null && value !== undefined) {
        columnTypes.set(columnName, inferSqlType(value));
      }
    }
  }

  return {
    columns,
    columnTypes,
    validColumns: new Set([SUBSET_COLUMN, ...columns])
  };
}

function quickSafetyCheck(condition) {
  const upper = condition.toUpperCase();

  if (condition.includes("--") || condition.includes("/*") || condition.includes("*/")) {
    throw new Error("SQL comments are not allowed in conditions");
  }

  if (condition.includes(";")) {
    throw new Error("Semicolons are not allowed in conditions");
  }

  for (const keyword of DANGEROUS_KEYWORDS) {
    if (new RegExp(`\\b${keyword}\\b`).test(upper)) {
      throw new Error(`SQL keyword '${keyword}' is not allowed in conditions`);
    }
  }
}

function tokenizeCondition(input) {
  const tokens = [];
  let position = 0;

  while (position < input.length) {
    const char = input[position];

    if (/\s/.test(char)) {
      position += 1;
      continue;
    }

    switch (char) {
      case "?":
        tokens.push({ type: "PLACEHOLDER" });
        position += 1;
        continue;
      case "(":
        tokens.push({ type: "LPAREN" });
        position += 1;
        continue;
      case ")":
        tokens.push({ type: "RPAREN" });
        position += 1;
        continue;
      case ",":
        tokens.push({ type: "COMMA" });
        position += 1;
        continue;
      case "=":
        tokens.push({ type: "EQ" });
        position += 1;
        continue;
      default:
        break;
    }

    const twoCharOperator = input.slice(position, position + 2);
    switch (twoCharOperator) {
      case "!=":
      case "<>":
        tokens.push({ type: "NE" });
        position += 2;
        continue;
      case "<=":
        tokens.push({ type: "LE" });
        position += 2;
        continue;
      case ">=":
        tokens.push({ type: "GE" });
        position += 2;
        continue;
      default:
        break;
    }

    switch (char) {
      case "<":
        tokens.push({ type: "LT" });
        position += 1;
        continue;
      case ">":
        tokens.push({ type: "GT" });
        position += 1;
        continue;
      default:
        break;
    }

    if (/[A-Za-z_]/.test(char)) {
      let end = position + 1;
      while (end < input.length && /[A-Za-z0-9_]/.test(input[end])) {
        end += 1;
      }
      const word = input.slice(position, end);
      const upperWord = word.toUpperCase();
      const keywordType = {
        AND: "AND",
        OR: "OR",
        NOT: "NOT",
        IS: "IS",
        NULL: "NULL",
        LIKE: "LIKE",
        REGEXP: "REGEXP",
        BETWEEN: "BETWEEN",
        IN: "IN"
      }[upperWord];

      tokens.push(keywordType ? { type: keywordType } : { type: "IDENTIFIER", value: word });
      position = end;
      continue;
    }

    if (char === '"') {
      const end = input.indexOf('"', position + 1);
      if (end === -1) {
        throw new Error("Unterminated quoted identifier");
      }

      tokens.push({
        type: "IDENTIFIER",
        value: input.slice(position + 1, end)
      });
      position = end + 1;
      continue;
    }

    throw new Error(`Unexpected character '${char}' in condition`);
  }

  tokens.push({ type: "EOF" });
  return tokens;
}

class ConditionValidator {
  constructor(tokens, validColumns) {
    this.tokens = tokens;
    this.position = 0;
    this.validColumns = new Set(
      [...validColumns].map((columnName) => columnName.toLowerCase()),
    );
  }

  current() {
    return this.tokens[this.position] ?? { type: "EOF" };
  }

  currentType() {
    return this.current().type;
  }

  advance() {
    if (this.position < this.tokens.length) {
      this.position += 1;
    }
  }

  expect(expectedType) {
    if (this.currentType() !== expectedType) {
      throw new Error(`Expected ${expectedType}, found ${this.currentType()}`);
    }

    this.advance();
  }

  validate() {
    this.parseExpr();

    if (this.currentType() !== "EOF") {
      throw new Error(`Unexpected token ${this.currentType()} after expression`);
    }
  }

  parseExpr() {
    this.parseAndExpr();
    while (this.currentType() === "OR") {
      this.advance();
      this.parseAndExpr();
    }
  }

  parseAndExpr() {
    this.parseUnaryExpr();
    while (this.currentType() === "AND") {
      this.advance();
      this.parseUnaryExpr();
    }
  }

  parseUnaryExpr() {
    if (this.currentType() === "NOT") {
      this.advance();
    }
    this.parsePrimaryExpr();
  }

  parsePrimaryExpr() {
    if (this.currentType() === "LPAREN") {
      this.advance();
      this.parseExpr();
      this.expect("RPAREN");
      return;
    }

    const current = this.current();
    if (current.type !== "IDENTIFIER") {
      throw new Error(`Expected column name, found ${current.type}`);
    }

    if (!this.validColumns.has(current.value.toLowerCase())) {
      throw new Error(`Unknown column '${current.value}' in condition`);
    }

    this.advance();

    switch (this.currentType()) {
      case "IS":
        this.advance();
        if (this.currentType() === "NOT") {
          this.advance();
        }
        this.expect("NULL");
        return;
      case "NOT":
        this.advance();
        switch (this.currentType()) {
          case "BETWEEN":
            this.advance();
            this.expect("PLACEHOLDER");
            this.expect("AND");
            this.expect("PLACEHOLDER");
            return;
          case "IN":
            this.advance();
            this.parseInList();
            return;
          case "LIKE":
          case "REGEXP":
            this.advance();
            this.expect("PLACEHOLDER");
            return;
          default:
            throw new Error(
              `Expected BETWEEN, IN, LIKE, or REGEXP after NOT, found ${this.currentType()}`,
            );
        }
      case "BETWEEN":
        this.advance();
        this.expect("PLACEHOLDER");
        this.expect("AND");
        this.expect("PLACEHOLDER");
        return;
      case "IN":
        this.advance();
        this.parseInList();
        return;
      case "LIKE":
      case "REGEXP":
        this.advance();
        this.expect("PLACEHOLDER");
        return;
      case "EQ":
      case "NE":
      case "LT":
      case "LE":
      case "GT":
      case "GE":
        this.advance();
        this.expect("PLACEHOLDER");
        return;
      default:
        throw new Error(
          `Expected operator after column name, found ${this.currentType()}`,
        );
    }
  }

  parseInList() {
    this.expect("LPAREN");
    this.expect("PLACEHOLDER");
    while (this.currentType() === "COMMA") {
      this.advance();
      this.expect("PLACEHOLDER");
    }
    this.expect("RPAREN");
  }
}

function validateCondition(condition, validColumns) {
  if (NUMERIC_EQUALITY_PATTERN.test(condition.trim())) {
    return;
  }

  quickSafetyCheck(condition);
  const validator = new ConditionValidator(tokenizeCondition(condition), validColumns);
  validator.validate();
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

function installRegexpFunction(db) {
  const regexCache = new Map();

  db.createFunction(
    "regexp",
    (_context, pattern, text) => {
      if (pattern === null || pattern === undefined || text === null || text === undefined) {
        return 0;
      }

      const cacheKey = String(pattern);
      let regex = regexCache.get(cacheKey);
      if (!regex) {
        if (cacheKey.length > 100_000) {
          throw new Error("REGEXP pattern is too large");
        }
        regex = new RegExp(cacheKey, "i");
        regexCache.set(cacheKey, regex);
      }

      return regex.test(String(text)) ? 1 : 0;
    },
    { deterministic: true, arity: 2 },
  );
}

function buildCreateMetadataSql(schema) {
  const columnDefs = [`"${SUBSET_COLUMN}" INTEGER PRIMARY KEY`];
  for (const columnName of schema.columns) {
    columnDefs.push(`"${columnName}" ${schema.columnTypes.get(columnName) ?? "TEXT"}`);
  }

  return `CREATE TABLE IF NOT EXISTS "${METADATA_TABLE}" (${columnDefs.join(", ")})`;
}

function buildInsertMetadataSql(columns) {
  const columnNames = [`"${SUBSET_COLUMN}"`, ...columns.map((column) => `"${column}"`)];
  const placeholders = Array.from({ length: columns.length + 1 }, () => "?");
  return `INSERT INTO "${METADATA_TABLE}"(${columnNames.join(", ")}) VALUES (${placeholders.join(", ")})`;
}

function ensureTables(db, tokenizer, schema) {
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

  db.exec(buildCreateMetadataSql(schema));

  db.exec({
    sql: `INSERT OR REPLACE INTO "${FTS_CONFIG_TABLE}"(key, value) VALUES ('tokenizer', ?)`,
    bind: [tokenizer],
  });

  installRegexpFunction(db);
}

function insertMetadata(db, metadata, columns) {
  db.exec("BEGIN");

  let keywordStatement;
  let metadataStatement;

  try {
    keywordStatement = db.prepare(
      `INSERT INTO "${FTS_TABLE}"(rowid, "${FTS_CONTENT_COLUMN}") VALUES (?, ?)`,
    );
    metadataStatement = db.prepare(buildInsertMetadataSql(columns));

    for (let documentId = 0; documentId < metadata.length; documentId += 1) {
      const value = metadata[documentId];
      keywordStatement.bind([documentId, metadataToText(value)]).stepReset();

      const rowObject = isPlainObject(value) ? value : null;
      const metadataRow = [documentId];
      for (const columnName of columns) {
        metadataRow.push(jsonToSql(rowObject?.[columnName]));
      }
      metadataStatement.bind(metadataRow).stepReset();
    }

    db.exec("COMMIT");
  } catch (error) {
    db.exec("ROLLBACK");
    throw error;
  } finally {
    metadataStatement?.finalize();
    keywordStatement?.finalize();
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

function execMetadataFilterQuery(db, condition, parameters, validColumns) {
  validateCondition(condition, validColumns);

  const rows = db.exec({
    sql: `SELECT "${SUBSET_COLUMN}" AS document_id FROM "${METADATA_TABLE}" WHERE ${condition}`,
    bind: parameters.map((value) => jsonToSql(value)),
    rowMode: "object",
    returnValue: "resultRows",
  });

  return rows.map((row) => Number(row.document_id));
}

class KeywordIndex {
  constructor(sqlite3, metadata, tokenizer) {
    this.sqlite3 = sqlite3;
    this.metadata = metadata ?? null;
    this.tokenizer = normalizeTokenizer(tokenizer);
    this.schema = collectMetadataSchema(metadata ?? []);
    this.db = new sqlite3.oo1.DB(":memory:", "c");
    ensureTables(this.db, this.tokenizer, this.schema);
    if (this.metadata) {
      insertMetadata(this.db, this.metadata, this.schema.columns);
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

  filterDocumentIds(condition, parameters = []) {
    if (!this.metadata) {
      throw new Error("metadata filtering requires metadata to be loaded for this index");
    }

    return execMetadataFilterQuery(this.db, condition, parameters, this.schema.validColumns);
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

  filterIndex({ name, condition, parameters = [] }) {
    const index = this.indices.get(name);
    if (!index) {
      throw new Error("metadata filtering requires metadata to be loaded for this index");
    }

    return index.filterDocumentIds(condition, parameters);
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
