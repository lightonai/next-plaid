import { expect, it } from "@effect/vitest";
import { Effect } from "effect";
import { vi } from "vitest";

import { DocumentTextDigestService } from "./document-text-digest-service.js";

it.effect("returns a typed failure when crypto.subtle.digest rejects", () =>
  Effect.gen(function*() {
    const originalCrypto = globalThis.crypto;
    const digest = vi.fn(async () => {
      throw new Error("boom");
    });

    Object.defineProperty(globalThis, "crypto", {
      configurable: true,
      value: {
        subtle: {
          digest,
        },
      },
    });

    yield* Effect.addFinalizer(() =>
      Effect.sync(() => {
        Object.defineProperty(globalThis, "crypto", {
          configurable: true,
          value: originalCrypto,
        });
      })
    );

    const service = yield* DocumentTextDigestService;
    const result = yield* Effect.result(service.sha256Hex("alpha semantic body"));

    expect(result._tag).toBe("Failure");
    if (result._tag !== "Failure") {
      throw new Error("expected digest rejection to fail with a typed error");
    }
    expect(result.failure._tag).toBe("PermanentClientError");
    expect(result.failure.cause).toBe("document_text_digest_failed");
    expect(digest).toHaveBeenCalledTimes(1);
  }).pipe(Effect.provide(DocumentTextDigestService.layer)),
);
