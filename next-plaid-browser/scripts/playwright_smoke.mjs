#!/usr/bin/env node

import { join } from "node:path";
import process from "node:process";

import {
  browserConfig,
  outputRoot,
  prepareHarness,
  startHarnessServer,
} from "./playwright_harness_common.mjs";

const requestedBrowser = process.argv[2] ?? "chromium";
const headless = process.env.PLAYWRIGHT_HEADLESS !== "0";

async function runSmoke(browserName) {
  const { installName, launcher, launchOptions } = browserConfig(browserName, headless);

  await prepareHarness(installName);

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
    const encodedQueryTopDocumentId = result.encodedSearch?.results?.[0]?.document_ids?.[0];
    const encodedQueryLayout = result.encodedQuery?.encoded?.payload?.layout;
    const encodedQueryEncoderId = result.encodedQuery?.encoded?.payload?.encoder?.encoder_id;
    const encoderBackend = result.encoderInit?.capabilities?.backend;
    const encoderPersistentStorage = result.encoderInit?.capabilities?.persistentStorage;
    const encoderState = result.encoderHealth?.state;
    const registerCorpusCreated = result.registerCorpus?.created;
    const syncCorpusChanged = result.syncCorpus?.sync?.changed;
    const mutableInitialTopDocumentId = result.mutableSearch?.results?.[0]?.document_ids?.[0];
    const mutableReloadedSyncChanged = result.mutableReloadedSync?.sync?.changed;
    const mutableReloadedSyncUnchanged = result.mutableReloadedSync?.sync?.unchanged;
    const mutableReloadedTopDocumentId =
      result.mutableReloadedSearch?.results?.[0]?.document_ids?.[0];
    const mutableCorpusLoaded = result.mutableCorpusState?.loaded;
    const mutableCorpusDocumentCount = result.mutableCorpusState?.summary?.document_count;
    const encoderEventStages = Array.isArray(result.encoderInitEvents)
      ? result.encoderInitEvents.map((event) => event?.stage)
      : [];
    const loadedIndices = result.health?.loaded_indices;
    const runtimeEncodedTopDocumentId =
      result.runtimeEncodedSearch?.results?.[0]?.document_ids?.[0];
    const sawAssetLoad =
      encoderEventStages.includes("asset_fetch_start") ||
      encoderEventStages.includes("asset_store_hit") ||
      encoderEventStages.includes("asset_memory_hit");

    if (
      installActivated !== true ||
      reloadedInitialIndices !== 0 ||
      storedLoadedDocuments !== 2 ||
      storedSemanticTopDocumentId !== 0 ||
      storedKeywordTopDocumentId !== 0 ||
      storedHybridTopDocumentId !== 1 ||
      storedFilteredKeywordTopDocumentId !== 1 ||
      registerCorpusCreated !== true ||
      syncCorpusChanged !== true ||
      mutableInitialTopDocumentId !== 0 ||
      mutableReloadedSyncChanged !== false ||
      mutableReloadedSyncUnchanged !== 2 ||
      mutableReloadedTopDocumentId !== 0 ||
      mutableCorpusLoaded !== true ||
      mutableCorpusDocumentCount !== 2 ||
      semanticTopDocumentId !== 0 ||
      keywordTopDocumentId !== 0 ||
      hybridTopDocumentId !== 1 ||
      filteredSemanticTopDocumentId !== 1 ||
      JSON.stringify(filteredKeywordDocumentIds) !== JSON.stringify([0, 2]) ||
      encodedQueryTopDocumentId !== 0 ||
      encodedQueryLayout !== "padded_query_length" ||
      encodedQueryEncoderId !== "tiny-encoder-proof" ||
      encoderBackend !== "wasm" ||
      encoderPersistentStorage !== true ||
      encoderState !== "ready" ||
      !sawAssetLoad ||
      !encoderEventStages.includes("session_create_complete") ||
      !encoderEventStages.includes("warmup_complete") ||
      !encoderEventStages.includes("ready") ||
      loadedIndices !== 3 ||
      runtimeEncodedTopDocumentId !== 0
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
