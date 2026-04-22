import { Context, Effect, Layer } from "effect";

import type { BrowserRuntimeError } from "./client-errors.js";
import { permanentClientError } from "./client-errors.js";

export interface DocumentTextDigestServiceApi {
  readonly sha256Hex: (text: string) => Effect.Effect<string, BrowserRuntimeError>;
}

export class DocumentTextDigestService
  extends Context.Service<DocumentTextDigestService, DocumentTextDigestServiceApi>()(
    "next-plaid-browser/DocumentTextDigestService",
  )
{
  static readonly layer = Layer.effect(DocumentTextDigestService)(
    makeDocumentTextDigestService(),
  );
}

const textEncoder = new TextEncoder();

function subtleUnavailableError(): BrowserRuntimeError {
  return permanentClientError({
    cause: "crypto_subtle_unavailable",
    message: "globalThis.crypto.subtle is not available for document text digesting",
    operation: "browser_runtime.document_text_digest",
    details: null,
  });
}

function digestFailedError(error: unknown): BrowserRuntimeError {
  return permanentClientError({
    cause: "document_text_digest_failed",
    message: "failed to compute SHA-256 digest for document text",
    operation: "browser_runtime.document_text_digest",
    details: error,
  });
}

function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes).map((value) => value.toString(16).padStart(2, "0")).join("");
}

function makeDocumentTextDigestService(): Effect.Effect<
  DocumentTextDigestServiceApi,
  never
> {
  return Effect.sync(() =>
    DocumentTextDigestService.of({
      sha256Hex: Effect.fn("DocumentTextDigestService.sha256Hex")(function*(text: string) {
        const subtle = globalThis.crypto?.subtle;
        if (subtle === undefined) {
          return yield* subtleUnavailableError();
        }

        const digest = yield* Effect.promise(() =>
          subtle.digest("SHA-256", textEncoder.encode(text))
        ).pipe(
          Effect.mapError(digestFailedError),
        );

        return bytesToHex(new Uint8Array(digest));
      }),
    }),
  );
}
