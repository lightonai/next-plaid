import {
  Cache as EffectCache,
  Context,
  Duration,
  Effect,
  Equal,
  Hash,
  Layer,
  Scope,
} from "effect";
import * as Exit from "effect/Exit";

import type { EncoderCapabilities } from "../model-worker/types.js";
import type { MatrixPayload } from "../generated/MatrixPayload.js";
import type { BrowserRuntimeError } from "./client-errors.js";
import { permanentClientError } from "./client-errors.js";
import { DocumentTextDigestService } from "./document-text-digest-service.js";
import { EncoderWorkerClient } from "./encoder-worker-client.js";

const DEFAULT_DOCUMENT_CACHE_CAPACITY = 128;
const INVALID_DOCUMENT_EMBEDDING_PAYLOAD_CAUSE = "invalid_document_embedding_payload";

export interface InvalidDocumentEmbeddingPayloadDetails {
  readonly rows: number;
  readonly dim: number;
  readonly valueCount: number;
}

export interface DocumentEmbeddingCacheServiceApi {
  readonly get: (args: {
    readonly capabilities: EncoderCapabilities;
    readonly text: string;
    readonly requestId?: string | undefined;
  }) => Effect.Effect<MatrixPayload, BrowserRuntimeError>;
  readonly clear: () => Effect.Effect<void>;
}

export class DocumentEmbeddingCacheService
  extends Context.Service<DocumentEmbeddingCacheService, DocumentEmbeddingCacheServiceApi>()(
    "next-plaid-browser/DocumentEmbeddingCacheService",
  )
{
  static readonly layer = (options: {
    readonly capacity?: number | undefined;
  } = {}) =>
    Layer.effect(DocumentEmbeddingCacheService)(
      makeDocumentEmbeddingCacheService(options),
    );
}

interface DocumentEmbeddingCacheKeyFields {
  readonly encoderId: string;
  readonly encoderBuild: string;
  readonly embeddingDim: number;
  readonly normalized: boolean;
  readonly textDigest: string;
  readonly text: string;
}

export class DocumentEmbeddingCacheKey implements Equal.Equal {
  readonly encoderId: string;
  readonly encoderBuild: string;
  readonly embeddingDim: number;
  readonly normalized: boolean;
  readonly textDigest: string;
  readonly text: string;
  private readonly hashValue: number;

  constructor(fields: DocumentEmbeddingCacheKeyFields) {
    this.encoderId = fields.encoderId;
    this.encoderBuild = fields.encoderBuild;
    this.embeddingDim = fields.embeddingDim;
    this.normalized = fields.normalized;
    this.textDigest = fields.textDigest;
    this.text = fields.text;

    let hashValue = Hash.hash(this.encoderId);
    hashValue = Hash.combine(hashValue, Hash.hash(this.encoderBuild));
    hashValue = Hash.combine(hashValue, Hash.hash(this.embeddingDim));
    hashValue = Hash.combine(hashValue, Hash.hash(this.normalized));
    hashValue = Hash.combine(hashValue, Hash.hash(this.textDigest));
    this.hashValue = Hash.optimize(hashValue);

    Object.freeze(this);
  }

  [Hash.symbol](): number {
    return this.hashValue;
  }

  [Equal.symbol](that: Equal.Equal): boolean {
    return that instanceof DocumentEmbeddingCacheKey &&
      this.encoderId === that.encoderId &&
      this.encoderBuild === that.encoderBuild &&
      this.embeddingDim === that.embeddingDim &&
      this.normalized === that.normalized &&
      this.textDigest === that.textDigest;
  }
}

function invalidDocumentEmbeddingPayloadError(
  details: InvalidDocumentEmbeddingPayloadDetails,
): BrowserRuntimeError {
  return permanentClientError({
    cause: INVALID_DOCUMENT_EMBEDDING_PAYLOAD_CAUSE,
    message: "document embedding payload is empty or malformed",
    operation: "browser_runtime.document_embedding_cache",
    details,
  });
}

function makeDocumentEmbeddingCacheKey(
  capabilities: EncoderCapabilities,
  text: string,
  textDigest: string,
): DocumentEmbeddingCacheKey {
  return new DocumentEmbeddingCacheKey({
    encoderId: capabilities.encoderId,
    encoderBuild: capabilities.encoderBuild,
    embeddingDim: capabilities.embeddingDim,
    normalized: capabilities.normalized,
    textDigest,
    text,
  });
}

function validatePayload(
  payload: MatrixPayload,
): Effect.Effect<MatrixPayload, BrowserRuntimeError> {
  const valueCount = payload.values.length;
  if (payload.rows <= 0 || payload.dim <= 0 || valueCount <= 0) {
    return Effect.fail(
      invalidDocumentEmbeddingPayloadError({
        rows: payload.rows,
        dim: payload.dim,
        valueCount,
      }),
    );
  }

  return Effect.succeed(payload);
}

export function getInvalidDocumentEmbeddingPayloadDetails(
  error: BrowserRuntimeError,
): InvalidDocumentEmbeddingPayloadDetails | null {
  if (error.cause !== INVALID_DOCUMENT_EMBEDDING_PAYLOAD_CAUSE) {
    return null;
  }

  const details = error.details;
  if (typeof details !== "object" || details === null) {
    return null;
  }

  const rows = Reflect.get(details, "rows");
  const dim = Reflect.get(details, "dim");
  const valueCount = Reflect.get(details, "valueCount");

  return typeof rows === "number" &&
      typeof dim === "number" &&
      typeof valueCount === "number"
    ? { rows, dim, valueCount }
    : null;
}

function makeDocumentEmbeddingCacheService(options: {
  readonly capacity?: number | undefined;
}): Effect.Effect<
  DocumentEmbeddingCacheServiceApi,
  never,
  EncoderWorkerClient | DocumentTextDigestService | Scope.Scope
> {
  return Effect.gen(function*() {
    const encoderClient = yield* EncoderWorkerClient;
    const digestService = yield* DocumentTextDigestService;
    const cache = yield* EffectCache.makeWith<
      DocumentEmbeddingCacheKey,
      MatrixPayload,
      BrowserRuntimeError
    >(
      (key) =>
        encoderClient.encodeDocument({ text: key.text }).pipe(
          Effect.map((encoded) => encoded.payload),
          Effect.flatMap(validatePayload),
        ),
      {
        capacity: options.capacity ?? DEFAULT_DOCUMENT_CACHE_CAPACITY,
        timeToLive: (exit) =>
          Exit.isSuccess(exit) ? Duration.infinity : Duration.zero,
      },
    );

    const clear = Effect.fn("DocumentEmbeddingCacheService.clear")(function*() {
      yield* EffectCache.invalidateAll(cache);
    });

    yield* Effect.addFinalizer(() => clear());

    return DocumentEmbeddingCacheService.of({
      get: Effect.fn("DocumentEmbeddingCacheService.get")(function*(args: {
        readonly capabilities: EncoderCapabilities;
        readonly text: string;
        readonly requestId?: string | undefined;
      }) {
        const textDigest = yield* digestService.sha256Hex(args.text);
        const key = makeDocumentEmbeddingCacheKey(
          args.capabilities,
          args.text,
          textDigest,
        );
        return yield* EffectCache.get(cache, key);
      }),
      clear,
    });
  });
}
