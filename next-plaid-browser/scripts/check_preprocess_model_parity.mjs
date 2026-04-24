#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const scriptDir = dirname(fileURLToPath(import.meta.url));
const browserRoot = resolve(scriptDir, "..");
const repoRoot = resolve(browserRoot, "..");
const targetRoot = join(browserRoot, "target", "preprocess-parity");
const modelCacheRoot = join(targetRoot, "models");
const casesPath = join(targetRoot, "cases.json");
const wasmOutDir = join(browserRoot, "target", "preprocess-parity-wasm");
const wasmJsPath = join(wasmOutDir, "next_plaid_browser_preprocess_wasm.js");

const MODEL_PRESETS = [
  {
    id: "mxbai-edge-colbert-v0-32m-onnx",
    modelId: "lightonai/mxbai-edge-colbert-v0-32m-onnx",
  },
  {
    id: "answerai-colbert-small-v1-onnx",
    modelId: "lightonai/answerai-colbert-small-v1-onnx",
  },
  {
    id: "GTE-ModernColBERT-v1",
    modelId: "lightonai/GTE-ModernColBERT-v1",
  },
];

const longEnglish = Array.from(
  { length: 80 },
  (_, index) => `Section ${index + 1} explains browser retrieval parity with native preprocessing.`,
).join(" ");

const edgeLength = Array.from({ length: 700 }, (_, index) => `edge${index}`).join(" ");

const CASES = [
  {
    id: "query-english",
    kind: "query",
    text: "How does ColBERT late interaction rank documents?",
  },
  {
    id: "document-english",
    kind: "document",
    text: "ColBERT ranks documents by comparing contextual token embeddings at query time.",
  },
  {
    id: "query-punctuation",
    kind: "query",
    text: "ACME, Inc. v2.0: alpha+beta/gamma? [draft]!",
  },
  {
    id: "document-punctuation",
    kind: "document",
    text: "The release notes mention C++17, Rust/WASM, file-name.ext, and '$special' symbols.",
  },
  {
    id: "query-long",
    kind: "query",
    text: longEnglish,
  },
  {
    id: "document-long",
    kind: "document",
    text: longEnglish,
  },
  {
    id: "query-cjk",
    kind: "query",
    text: "検索システムは中文和日本語の入力を同じように処理できますか",
  },
  {
    id: "document-cjk",
    kind: "document",
    text: "浏览器运行时需要正确处理中文、日本語、한국어和混合 English text.",
  },
  {
    id: "query-emoji",
    kind: "query",
    text: "Can the encoder handle emoji like 🚀, 🧪, and ✅ without drift?",
  },
  {
    id: "document-emoji",
    kind: "document",
    text: "Emoji-heavy notes include status ✅, experiment 🧪, launch 🚀, and fallback text.",
  },
  {
    id: "query-edge-length",
    kind: "query",
    text: edgeLength,
  },
  {
    id: "document-edge-length",
    kind: "document",
    text: edgeLength,
  },
];

function safeModelDir(modelId) {
  return modelId.replaceAll("/", "__");
}

function hfResolveUrl(modelId, fileName) {
  return `https://huggingface.co/${modelId}/resolve/main/${fileName}`;
}

function run(command, args, cwd) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`,
      },
      stdio: "inherit",
    });

    child.on("exit", (code) => {
      if (code === 0) {
        resolvePromise();
      } else {
        rejectPromise(new Error(`${command} exited with code ${code}`));
      }
    });
    child.on("error", rejectPromise);
  });
}

function runCapture(command, args, cwd) {
  return new Promise((resolvePromise, rejectPromise) => {
    const stdout = [];
    const child = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`,
      },
      stdio: ["ignore", "pipe", "inherit"],
    });

    child.stdout.on("data", (chunk) => stdout.push(chunk));
    child.on("exit", (code) => {
      if (code === 0) {
        resolvePromise(Buffer.concat(stdout).toString("utf8"));
      } else {
        rejectPromise(new Error(`${command} exited with code ${code}`));
      }
    });
    child.on("error", rejectPromise);
  });
}

async function downloadIfMissing(modelId, fileName, outputPath) {
  if (existsSync(outputPath)) {
    return;
  }

  const response = await fetch(hfResolveUrl(modelId, fileName), {
    headers: {
      "user-agent": "next-plaid-browser-parity/0.1 (https://github.com/lightonai/next-plaid)",
      accept: "application/octet-stream",
    },
  });

  if (!response.ok) {
    throw new Error(
      `failed to download ${fileName} for ${modelId}: HTTP ${response.status}`,
    );
  }

  await writeFile(outputPath, Buffer.from(await response.arrayBuffer()));
}

async function prepareCases() {
  await mkdir(targetRoot, { recursive: true });
  await writeFile(casesPath, `${JSON.stringify({ cases: CASES }, null, 2)}\n`);
}

async function prepareModel(model) {
  const modelDir = join(modelCacheRoot, safeModelDir(model.modelId));
  await mkdir(modelDir, { recursive: true });
  const tokenizerPath = join(modelDir, "tokenizer.json");
  const configPath = join(modelDir, "onnx_config.json");

  await downloadIfMissing(model.modelId, "tokenizer.json", tokenizerPath);
  await downloadIfMissing(model.modelId, "onnx_config.json", configPath);

  return { tokenizerPath, configPath };
}

async function buildWasmPreprocessor() {
  await run(
    "wasm-pack",
    [
      "build",
      "--target",
      "nodejs",
      "--dev",
      "--out-dir",
      "../../target/preprocess-parity-wasm",
    ],
    join(browserRoot, "crates", "next-plaid-browser-preprocess-wasm"),
  );
}

async function nativeExpected(tokenizerPath, configPath) {
  const stdout = await runCapture(
    "cargo",
    [
      "run",
      "--quiet",
      "--manifest-path",
      join(repoRoot, "Cargo.toml"),
      "-p",
      "next-plaid-preprocess",
      "--bin",
      "preprocess-parity-oracle",
      "--features",
      "parity-oracle",
      "--",
      tokenizerPath,
      configPath,
      casesPath,
    ],
    repoRoot,
  );

  return JSON.parse(stdout);
}

function bytes(buffer) {
  return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
}

function normalizePrepared(value) {
  return {
    input_ids: Array.from(value.input_ids ?? []),
    attention_mask: Array.from(value.attention_mask ?? []),
    token_type_ids:
      value.token_type_ids === undefined || value.token_type_ids === null
        ? null
        : Array.from(value.token_type_ids),
    retain_row_indices:
      value.retain_row_indices === undefined || value.retain_row_indices === null
        ? null
        : Array.from(value.retain_row_indices),
    active_length: value.active_length,
  };
}

function expectedShape(expectedCase) {
  return {
    input_ids: expectedCase.input_ids,
    attention_mask: expectedCase.attention_mask,
    token_type_ids: expectedCase.token_type_ids ?? null,
    retain_row_indices: expectedCase.retain_row_indices ?? null,
    active_length: expectedCase.active_length,
  };
}

function assertEqual(modelId, caseId, field, actual, expected) {
  const actualJson = JSON.stringify(actual);
  const expectedJson = JSON.stringify(expected);
  if (actualJson !== expectedJson) {
    throw new Error(
      `${modelId} ${caseId} ${field} mismatch\nexpected: ${expectedJson}\nactual:   ${actualJson}`,
    );
  }
}

async function browserActual(wasm, tokenizerPath, configPath) {
  const tokenizerBytes = bytes(await readFile(tokenizerPath));
  const configBytes = bytes(await readFile(configPath));
  wasm.reset();
  wasm.init(tokenizerBytes, configBytes);

  return CASES.map((testCase) => {
    const prepared =
      testCase.kind === "query"
        ? wasm.prepare_query(testCase.text)
        : wasm.prepare_document(testCase.text);
    return {
      id: testCase.id,
      kind: testCase.kind,
      ...normalizePrepared(prepared),
    };
  });
}

async function checkModel(wasm, model) {
  const { tokenizerPath, configPath } = await prepareModel(model);
  const expected = await nativeExpected(tokenizerPath, configPath);
  const actualCases = await browserActual(wasm, tokenizerPath, configPath);
  const expectedById = new Map(expected.cases.map((testCase) => [testCase.id, testCase]));

  for (const actualCase of actualCases) {
    const expectedCase = expectedById.get(actualCase.id);
    if (expectedCase === undefined) {
      throw new Error(`${model.modelId} missing native expected case ${actualCase.id}`);
    }

    const actualShape = expectedShape(actualCase);
    const expectedCaseShape = expectedShape(expectedCase);
    assertEqual(model.modelId, actualCase.id, "prepared buffers", actualShape, expectedCaseShape);
  }

  console.log(`${model.modelId}: ${actualCases.length} preprocessing cases matched`);
}

async function main() {
  await prepareCases();
  await buildWasmPreprocessor();
  const wasm = require(wasmJsPath);

  for (const model of MODEL_PRESETS) {
    await checkModel(wasm, model);
  }

  console.log(
    JSON.stringify(
      {
        models: MODEL_PRESETS.map((model) => model.modelId),
        cases: CASES.map((testCase) => testCase.id),
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
