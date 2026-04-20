import { Schema } from "effect";

export class TransientClientError extends Schema.TaggedErrorClass<TransientClientError>()(
  "TransientClientError",
  {
    cause: Schema.String,
    message: Schema.String,
    operation: Schema.String,
    requestId: Schema.NullOr(Schema.String),
    details: Schema.Unknown,
  },
) {}

export class PermanentClientError extends Schema.TaggedErrorClass<PermanentClientError>()(
  "PermanentClientError",
  {
    cause: Schema.String,
    message: Schema.String,
    operation: Schema.String,
    requestId: Schema.NullOr(Schema.String),
    details: Schema.Unknown,
  },
) {}

export class DegradedClientError extends Schema.TaggedErrorClass<DegradedClientError>()(
  "DegradedClientError",
  {
    cause: Schema.String,
    message: Schema.String,
    operation: Schema.String,
    requestId: Schema.NullOr(Schema.String),
    details: Schema.Unknown,
  },
) {}

export type SearchClientError =
  | TransientClientError
  | PermanentClientError
  | DegradedClientError;

export type EncoderClientError =
  | TransientClientError
  | PermanentClientError
  | DegradedClientError;

export type BrowserRuntimeError =
  | TransientClientError
  | PermanentClientError
  | DegradedClientError;

interface ClientErrorFields {
  cause: string;
  message: string;
  operation: string;
  requestId?: string | null;
  details?: unknown;
}

export function transientClientError(fields: ClientErrorFields): TransientClientError {
  return new TransientClientError({
    cause: fields.cause,
    message: fields.message,
    operation: fields.operation,
    requestId: fields.requestId ?? null,
    details: fields.details ?? null,
  });
}

export function permanentClientError(fields: ClientErrorFields): PermanentClientError {
  return new PermanentClientError({
    cause: fields.cause,
    message: fields.message,
    operation: fields.operation,
    requestId: fields.requestId ?? null,
    details: fields.details ?? null,
  });
}

export function degradedClientError(fields: ClientErrorFields): DegradedClientError {
  return new DegradedClientError({
    cause: fields.cause,
    message: fields.message,
    operation: fields.operation,
    requestId: fields.requestId ?? null,
    details: fields.details ?? null,
  });
}

const fatalWorkerCauses = new Set([
  "timeout",
  "worker_crashed",
  "worker_messageerror",
  "worker_disposed",
  "worker_spawn_failed",
  "malformed_envelope",
  "unknown_request_id",
  "worker_failure_envelope",
  "worker_init_failed",
  "worker_encode_failed",
  "send_failed",
]);

export function isClientError(
  error: unknown,
): error is TransientClientError | PermanentClientError | DegradedClientError {
  return (
    error instanceof TransientClientError ||
    error instanceof PermanentClientError ||
    error instanceof DegradedClientError
  );
}

export function isFatalWorkerError(
  error: SearchClientError | EncoderClientError,
): boolean {
  return fatalWorkerCauses.has(error.cause);
}
