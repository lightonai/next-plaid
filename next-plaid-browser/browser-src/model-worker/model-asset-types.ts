import { Effect } from "effect";

import {
  WorkerRuntimeError,
  type WorkerRuntimeError as WorkerRuntimeErrorType,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import type { EncoderIdentity } from "../shared/search-contract.js";

export type ModelAssetKind = "model" | "tokenizer" | "onnxConfig";
export type DurableModelAssetStoreKind = "opfs" | "indexeddb";
export type ModelAssetStoreKind =
  | DurableModelAssetStoreKind
  | "transient";

export interface ModelAssetIdentityInput {
  readonly encoder: EncoderIdentity;
  readonly modelUrl: string;
  readonly tokenizerUrl: string;
  readonly onnxConfigUrl: string;
}

export interface ModelAssetPackageKey {
  readonly packageId: string;
  readonly encoderId: string;
  readonly encoderBuild: string;
  readonly embeddingDim: number;
  readonly normalized: boolean;
  readonly modelUrl: string;
  readonly tokenizerUrl: string;
  readonly onnxConfigUrl: string;
}

export interface ModelAssetPackage {
  readonly key: ModelAssetPackageKey;
  readonly modelBytes: Uint8Array;
  readonly tokenizerBytes: Uint8Array;
  readonly onnxConfigBytes: Uint8Array;
}

export interface ModelAssetEntry {
  readonly kind: ModelAssetKind;
  readonly url: string;
  readonly fileName: string;
}

const textEncoder = new TextEncoder();

export const MODEL_ASSET_DB_NAME = "next-plaid-browser-model-assets";
export const MODEL_ASSET_OPFS_ROOT_DIR = "next-plaid-browser-model-assets";
export const MODEL_ASSET_KINDS: ReadonlyArray<ModelAssetKind> = [
  "model",
  "tokenizer",
  "onnxConfig",
];

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join(
    "",
  );
}

export function makeModelAssetEntries(
  key: ModelAssetPackageKey,
): ReadonlyArray<ModelAssetEntry> {
  return [
    {
      kind: "model",
      url: key.modelUrl,
      fileName: "model.onnx",
    },
    {
      kind: "tokenizer",
      url: key.tokenizerUrl,
      fileName: "tokenizer.json",
    },
    {
      kind: "onnxConfig",
      url: key.onnxConfigUrl,
      fileName: "onnx_config.json",
    },
  ];
}

export function modelAssetBytes(
  pkg: ModelAssetPackage,
  kind: ModelAssetKind,
): Uint8Array {
  switch (kind) {
    case "model":
      return pkg.modelBytes;
    case "tokenizer":
      return pkg.tokenizerBytes;
    case "onnxConfig":
      return pkg.onnxConfigBytes;
  }
}

export function supportsOpfs(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.storage?.getDirectory === "function"
  );
}

export function supportsIndexedDb(): boolean {
  return (
    typeof globalThis.indexedDB !== "undefined" &&
    typeof globalThis.IDBKeyRange !== "undefined"
  );
}

export function makeModelAssetPackageKey(
  input: ModelAssetIdentityInput,
): Effect.Effect<ModelAssetPackageKey, WorkerRuntimeErrorType> {
  const payload = JSON.stringify({
    encoder_id: input.encoder.encoder_id,
    encoder_build: input.encoder.encoder_build,
    embedding_dim: input.encoder.embedding_dim,
    normalized: input.encoder.normalized,
    model_url: input.modelUrl,
    tokenizer_url: input.tokenizerUrl,
    onnx_config_url: input.onnxConfigUrl,
  });

  return Effect.tryPromise({
    try: async () => {
      if (typeof globalThis.crypto?.subtle?.digest !== "function") {
        throw workerRuntimeError({
          operation: "model_asset_store.package_id",
          message: "crypto.subtle.digest is unavailable in this worker",
        });
      }

      const digest = await globalThis.crypto.subtle.digest(
        "SHA-256",
        textEncoder.encode(payload),
      );

      return {
        packageId: bytesToHex(new Uint8Array(digest)),
        encoderId: input.encoder.encoder_id,
        encoderBuild: input.encoder.encoder_build,
        embeddingDim: input.encoder.embedding_dim,
        normalized: input.encoder.normalized,
        modelUrl: input.modelUrl,
        tokenizerUrl: input.tokenizerUrl,
        onnxConfigUrl: input.onnxConfigUrl,
      } satisfies ModelAssetPackageKey;
    },
    catch: (error) =>
      error instanceof WorkerRuntimeError
        ? error
        : workerRuntimeErrorFromUnknown(
            "model_asset_store.package_id",
            error,
            "failed to compute model asset package id",
          ),
  });
}
