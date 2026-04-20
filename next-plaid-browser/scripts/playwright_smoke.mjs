#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { mkdir, readFile } from "node:fs/promises";
import { extname, join, normalize, relative, resolve } from "node:path";
import process from "node:process";
import { chromium, firefox, webkit } from "playwright";

const workspaceRoot = resolve(import.meta.dirname, "..");
const outputRoot = join(workspaceRoot, "output", "playwright");

const requestedBrowser = process.argv[2] ?? "chromium";
const headless = process.env.PLAYWRIGHT_HEADLESS !== "0";
const llvmClang = "/opt/homebrew/opt/llvm/bin/clang";

const mimeTypes = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
};

const crossOriginHeaders = {
  "cross-origin-opener-policy": "same-origin",
  "cross-origin-embedder-policy": "require-corp",
  "cross-origin-resource-policy": "same-origin",
};

function browserConfig(name) {
  switch (name) {
    case "chromium":
      return { installName: "chromium", launcher: chromium, launchOptions: { headless } };
    case "chrome":
      return {
        installName: "chrome",
        launcher: chromium,
        launchOptions: { channel: "chrome", headless },
      };
    case "firefox":
      return { installName: "firefox", launcher: firefox, launchOptions: { headless } };
    case "webkit":
      return { installName: "webkit", launcher: webkit, launchOptions: { headless } };
    default:
      throw new Error(`unsupported browser: ${name}`);
  }
}

function runCommand(command, args, cwd) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd,
      env: {
        ...process.env,
        PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`,
        ...(existsSync(llvmClang)
          ? { CC_wasm32_unknown_unknown: process.env.CC_wasm32_unknown_unknown ?? llvmClang }
          : {}),
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

async function staticFileResponse(urlPath) {
  if (urlPath === "/favicon.ico") {
    return {
      body: Buffer.alloc(0),
      contentType: "image/x-icon",
      statusCode: 204,
    };
  }

  const relativePath = urlPath.endsWith("/") ? `${urlPath}index.html` : urlPath;
  const normalized = normalize(relativePath).replace(/^[/\\]+/, "");
  const fullPath = resolve(workspaceRoot, normalized);
  const relativeToRoot = relative(workspaceRoot, fullPath);

  if (relativeToRoot.startsWith("..") || relativeToRoot === "") {
    throw new Error("path escapes workspace root");
  }

  const body = await readFile(fullPath);
  return {
    body,
    contentType: mimeTypes[extname(fullPath)] ?? "application/octet-stream",
    statusCode: 200,
  };
}

function startHarnessServer() {
  const server = createServer(async (request, response) => {
    try {
      const url = new URL(request.url ?? "/", "http://127.0.0.1");
      const { body, contentType, statusCode } = await staticFileResponse(url.pathname);
      response.writeHead(statusCode, {
        "content-type": contentType,
        ...crossOriginHeaders,
      });
      response.end(body);
    } catch {
      response.writeHead(404, {
        "content-type": "text/plain; charset=utf-8",
        ...crossOriginHeaders,
      });
      response.end("not found");
    }
  });

  return new Promise((resolvePromise) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (address && typeof address === "object") {
        resolvePromise({
          server,
          url: `http://127.0.0.1:${address.port}/playwright-harness/`,
        });
      }
    });
  });
}

async function buildWasmHarness() {
  await runCommand(
    "wasm-pack",
    [
      "build",
      "--target",
      "web",
      "--dev",
      "--out-dir",
      "../../playwright-harness/pkg",
    ],
    join(workspaceRoot, "crates", "next-plaid-browser-wasm"),
  );
}

async function ensurePlaywrightBrowser(installName) {
  if (installName === "chrome") {
    return;
  }
  await runCommand("npx", ["playwright", "install", installName], workspaceRoot);
}

async function runSmoke(browserName) {
  const { installName, launcher, launchOptions } = browserConfig(browserName);

  await mkdir(outputRoot, { recursive: true });
  await buildWasmHarness();
  await ensurePlaywrightBrowser(installName);

  const { server, url } = await startHarnessServer();
  const browser = await launcher.launch(launchOptions);

  try {
    const page = await browser.newPage();
    page.on("console", (message) => {
      console.log(`[browser:${message.type()}] ${message.text()}`);
    });
    page.on("pageerror", (error) => {
      console.error(`[browser:pageerror] ${error.stack ?? error.message}`);
    });
    await page.goto(url, { waitUntil: "networkidle" });
    await page.locator("#status[data-state='ok']").waitFor({ timeout: 15000 });
    const payload = await page.locator("#status").textContent();
    const result = JSON.parse(payload ?? "{}");
    const installActivated = result.installBundle?.activated;
    const reloadedInitialIndices = result.reloadedInitialHealth?.loaded_indices;
    const storedLoadedDocuments = result.loadStoredBundle?.summary?.num_documents;
    const storedSemanticTopDocumentId = result.storedSemanticSearch?.results?.[0]?.document_ids?.[0];
    const storedKeywordTopDocumentId = result.storedKeywordSearch?.results?.[0]?.document_ids?.[0];
    const storedHybridTopDocumentId = result.storedHybridSearch?.results?.[0]?.document_ids?.[0];
    const storedFilteredKeywordTopDocumentId =
      result.storedFilteredKeywordSearch?.results?.[0]?.document_ids?.[0];
    const semanticTopDocumentId = result.semanticSearch?.results?.[0]?.document_ids?.[0];
    const keywordTopDocumentId = result.keywordSearch?.results?.[0]?.document_ids?.[0];
    const hybridTopDocumentId = result.hybridSearch?.results?.[0]?.document_ids?.[0];
    const filteredSemanticTopDocumentId =
      result.filteredSemanticSearch?.results?.[0]?.document_ids?.[0];
    const filteredKeywordDocumentIds =
      result.filteredKeywordSearch?.results?.[0]?.document_ids ?? [];
    const loadedIndices = result.health?.loaded_indices;

    if (
      installActivated !== true ||
      reloadedInitialIndices !== 0 ||
      storedLoadedDocuments !== 2 ||
      storedSemanticTopDocumentId !== 0 ||
      storedKeywordTopDocumentId !== 0 ||
      storedHybridTopDocumentId !== 1 ||
      storedFilteredKeywordTopDocumentId !== 1 ||
      semanticTopDocumentId !== 0 ||
      keywordTopDocumentId !== 0 ||
      hybridTopDocumentId !== 1 ||
      filteredSemanticTopDocumentId !== 1 ||
      JSON.stringify(filteredKeywordDocumentIds) !== JSON.stringify([0, 2]) ||
      loadedIndices !== 2
    ) {
      throw new Error(`unexpected smoke result: ${JSON.stringify(result)}`);
    }

    const screenshotPath = join(outputRoot, `smoke-${browserName}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });

    console.log(
      JSON.stringify(
        {
          browser: browserName,
          url,
          screenshot: screenshotPath,
          result,
        },
        null,
        2,
      ),
    );
  } finally {
    await browser.close();
    server.close();
  }
}

runSmoke(requestedBrowser).catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
