#!/usr/bin/env node

import { build } from "esbuild";
import { cp, mkdir, rm } from "node:fs/promises";
import { spawn } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const workspaceRoot = resolve(scriptDir, "..");
const srcRoot = join(workspaceRoot, "browser-src");
const harnessRoot = join(workspaceRoot, "playwright-harness");
const ortDistRoot = join(workspaceRoot, "node_modules", "onnxruntime-web", "dist");
const ortOutputRoot = join(harnessRoot, "ort");
const generatedTypesRoot = join(srcRoot, "generated");

function runCommand(command, args) {
  return new Promise((resolvePromise, rejectPromise) => {
    const child = spawn(command, args, {
      cwd: workspaceRoot,
      env: {
        ...process.env,
        TS_RS_EXPORT_DIR: generatedTypesRoot,
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

async function bundle(entryPoint, outfile) {
  await mkdir(dirname(outfile), { recursive: true });
  await build({
    entryPoints: [entryPoint],
    outfile,
    bundle: true,
    format: "esm",
    platform: "browser",
    target: "es2022",
    sourcemap: false,
    logLevel: "info",
  });
}

async function main() {
  await mkdir(harnessRoot, { recursive: true });
  await rm(generatedTypesRoot, { recursive: true, force: true });
  await mkdir(generatedTypesRoot, { recursive: true });
  await runCommand("cargo", [
    "run",
    "--manifest-path",
    join(workspaceRoot, "Cargo.toml"),
    "-p",
    "next-plaid-browser-contract",
    "--bin",
    "export_bindings",
  ]);
  await bundle(join(srcRoot, "playwright-harness", "app.ts"), join(harnessRoot, "app.js"));
  await bundle(
    join(srcRoot, "model-worker", "encoder-worker.ts"),
    join(harnessRoot, "encoder-worker.js"),
  );

  await mkdir(ortOutputRoot, { recursive: true });
  await cp(ortDistRoot, ortOutputRoot, { recursive: true, force: true });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
});
