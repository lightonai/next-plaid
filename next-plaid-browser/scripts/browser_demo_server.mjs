#!/usr/bin/env node

import process from "node:process";

import {
  buildBrowserHarness,
  buildWasmHarness,
  startHarnessServer,
} from "./playwright_harness_common.mjs";

async function main() {
  await buildWasmHarness();
  await buildBrowserHarness();

  const { server, url } = await startHarnessServer();
  const demoUrl = new URL(url);
  demoUrl.searchParams.set("scenario", "interactive-demo");

  console.log(`Next Plaid browser demo ready at ${demoUrl.toString()}`);
  console.log("Press Ctrl+C to stop the demo server.");

  const close = () =>
    new Promise((resolve) => {
      server.close(() => resolve());
    });

  process.on("SIGINT", () => {
    void close().then(() => {
      process.exit(0);
    });
  });

  process.on("SIGTERM", () => {
    void close().then(() => {
      process.exit(0);
    });
  });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
