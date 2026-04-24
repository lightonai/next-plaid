import { Schema } from "effect";

export class WorkerRuntimeError extends Schema.TaggedErrorClass<WorkerRuntimeError>()(
  "WorkerRuntimeError",
  {
    operation: Schema.String,
    message: Schema.String,
    details: Schema.Unknown,
  },
) {}

interface WorkerRuntimeErrorFields {
  readonly operation: string;
  readonly message: string;
  readonly details?: unknown;
}

export function workerRuntimeError(
  fields: WorkerRuntimeErrorFields,
): WorkerRuntimeError {
  return new WorkerRuntimeError({
    operation: fields.operation,
    message: fields.message,
    details: fields.details ?? null,
  });
}

export function workerRuntimeErrorFromUnknown(
  operation: string,
  error: unknown,
  prefix?: string,
): WorkerRuntimeError {
  const rendered =
    error instanceof Error ? error.stack ?? error.message : String(error);
  return workerRuntimeError({
    operation,
    message: prefix ? `${prefix}: ${rendered}` : rendered,
    details: error,
  });
}
