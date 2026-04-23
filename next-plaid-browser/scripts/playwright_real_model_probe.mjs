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

const DEFAULT_MODEL_PRESETS = [
  "mxbai-edge-colbert-v0-32m-onnx",
  "answerai-colbert-small-v1-onnx",
];

const ALL_MODEL_PRESETS = [
  ...DEFAULT_MODEL_PRESETS,
  "GTE-ModernColBERT-v1",
];

const EXPECTED_CAPABILITIES = {
  "answerai-colbert-small-v1-onnx": {
    embeddingDim: 96,
    queryLength: 32,
    documentLength: 300,
    usesTokenTypeIds: true,
    doLowerCase: false,
    doQueryExpansion: false,
    queryOutputLayout: "ragged",
  },
  "mxbai-edge-colbert-v0-32m-onnx": {
    embeddingDim: 64,
    queryLength: 48,
    documentLength: 512,
    usesTokenTypeIds: false,
    doLowerCase: true,
    doQueryExpansion: false,
    queryOutputLayout: "ragged",
  },
};

function selectedModelPresets(args) {
  const filtered = args.filter((value) => !value.startsWith("--"));
  if (args.includes("--all")) {
    return ALL_MODEL_PRESETS;
  }
  return filtered.length > 0 ? filtered : DEFAULT_MODEL_PRESETS;
}

function eventStages(events) {
  return Array.isArray(events)
    ? events
      .map((event) => event?.stage)
      .filter((stage) => typeof stage === "string")
    : [];
}

function eventDuration(events, stage) {
  if (!Array.isArray(events)) {
    return null;
  }
  const match = events.find((event) => event?.stage === stage);
  return typeof match?.durationMs === "number" ? match.durationMs : null;
}

function assertPhaseSearches(phase, phaseName) {
  if (!Array.isArray(phase?.searches) || phase.searches.length === 0) {
    throw new Error(`${phaseName} returned no search summaries`);
  }

  for (const search of phase.searches) {
    const returnedSlugs = Array.isArray(search?.returnedSlugs) ? search.returnedSlugs : [];
    if (returnedSlugs[0] !== search?.expectedSlug) {
      throw new Error(
        `${phaseName} query "${search?.queryId ?? "unknown"}" did not rank expected slug first ` +
        `"${search?.expectedSlug ?? "unknown"}"; got ${JSON.stringify(returnedSlugs)}`,
      );
    }
  }
}

function assertCapabilities(capabilities, expectedPreset) {
  const expected = EXPECTED_CAPABILITIES[expectedPreset];
  if (expected === undefined) {
    return;
  }

  for (const [key, expectedValue] of Object.entries(expected)) {
    if (capabilities?.[key] !== expectedValue) {
      throw new Error(
        `expected ${expectedPreset} capability ${key}=${JSON.stringify(expectedValue)} ` +
        `but got ${JSON.stringify(capabilities?.[key])}`,
      );
    }
  }
}

function validateProbeResult(result, expectedPreset) {
  if (result?.scenario !== "real-model-probe") {
    throw new Error(`unexpected probe scenario: ${JSON.stringify(result?.scenario)}`);
  }

  if (result?.modelPreset?.id !== expectedPreset) {
    throw new Error(
      `expected preset "${expectedPreset}" but got ${JSON.stringify(result?.modelPreset?.id)}`,
    );
  }

  if (result?.initialPhase?.encoderCapabilities?.backend !== "wasm") {
    throw new Error(`encoder backend was not wasm: ${JSON.stringify(result?.initialPhase)}`);
  }
  assertCapabilities(result?.initialPhase?.encoderCapabilities, expectedPreset);
  assertCapabilities(result?.reloadedPhase?.encoderCapabilities, expectedPreset);

  if (result?.initialPhase?.syncCorpus?.summary?.has_dense_state !== true) {
    throw new Error(`initial sync did not produce dense state: ${JSON.stringify(result?.initialPhase?.syncCorpus)}`);
  }

  if (result?.reloadedPhase?.syncCorpus?.sync?.changed !== false) {
    throw new Error(`reload sync was not a no-op: ${JSON.stringify(result?.reloadedPhase?.syncCorpus)}`);
  }

  if (result?.reloadedPhase?.mutableCorpusState?.loaded !== true) {
    throw new Error(`reloaded mutable corpus was not marked loaded: ${JSON.stringify(result?.reloadedPhase?.mutableCorpusState)}`);
  }

  const initialStages = eventStages(result?.initialPhase?.encoderInitEvents);
  const reloadedStages = eventStages(result?.reloadedPhase?.encoderInitEvents);

  for (const required of ["session_create_complete", "warmup_complete", "ready"]) {
    if (!initialStages.includes(required)) {
      throw new Error(`initial encoder events missing ${required}: ${JSON.stringify(initialStages)}`);
    }
    if (!reloadedStages.includes(required)) {
      throw new Error(`reloaded encoder events missing ${required}: ${JSON.stringify(reloadedStages)}`);
    }
  }

  assertPhaseSearches(result?.initialPhase, "initialPhase");
  assertPhaseSearches(result?.reloadedPhase, "reloadedPhase");
}

function probeUrl(baseUrl, modelPreset) {
  const url = new URL(baseUrl);
  url.searchParams.set("scenario", "real-model-probe");
  url.searchParams.set("modelPreset", modelPreset);
  url.searchParams.set("corpusPreset", "next-plaid-docs-v1");
  return url.toString();
}

async function runProbe(browserName, modelPresets) {
  const { installName, launcher, launchOptions } = browserConfig(browserName, headless);

  await prepareHarness(installName);

  const { server, url } = await startHarnessServer();
  const browser = await launcher.launch(launchOptions);

  try {
    const summaries = [];
    for (const modelPreset of modelPresets) {
      const page = await browser.newPage();
      page.on("console", (message) => {
        console.log(`[browser:${modelPreset}:${message.type()}] ${message.text()}`);
      });
      page.on("pageerror", (error) => {
        console.error(
          `[browser:${modelPreset}:pageerror] ${error.stack ?? error.message}`,
        );
      });

      try {
        await page.goto(probeUrl(url, modelPreset), { waitUntil: "networkidle" });
        await page.waitForFunction(() => {
          const node = document.getElementById("status");
          return node?.dataset.state === "ok" || node?.dataset.state === "error";
        }, { timeout: 300000 });

        const state = await page.locator("#status").evaluate((node) =>
          node instanceof HTMLElement ? node.dataset.state ?? null : null
        );
        if (state !== "ok") {
          const errorText = await page.locator("#status").textContent();
          throw new Error(
            `real-model probe failed for ${modelPreset}: ${errorText ?? "unknown page error"}`,
          );
        }

        const payload = await page.locator("#status").textContent();
        const result = JSON.parse(payload ?? "{}");
        validateProbeResult(result, modelPreset);

        const screenshotPath = join(outputRoot, `real-model-${browserName}-${modelPreset}.png`);
        await page.screenshot({ path: screenshotPath, fullPage: true });

        summaries.push({
          modelPreset,
          screenshot: screenshotPath,
          encoderBackend: result.initialPhase.encoderCapabilities.backend,
          persistentStorage: result.initialPhase.encoderCapabilities.persistentStorage,
          initialSessionCreateMs: eventDuration(
            result.initialPhase.encoderInitEvents,
            "session_create_complete",
          ),
          initialWarmupMs: eventDuration(
            result.initialPhase.encoderInitEvents,
            "warmup_complete",
          ),
          reloadedSessionCreateMs: eventDuration(
            result.reloadedPhase.encoderInitEvents,
            "session_create_complete",
          ),
          reloadedWarmupMs: eventDuration(
            result.reloadedPhase.encoderInitEvents,
            "warmup_complete",
          ),
          initialSearches: result.initialPhase.searches,
          reloadedSearches: result.reloadedPhase.searches,
        });
      } finally {
        await page.close();
      }
    }

    console.log(
      JSON.stringify(
        {
          browser: browserName,
          url,
          models: summaries,
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

runProbe(requestedBrowser, selectedModelPresets(process.argv.slice(3))).catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
