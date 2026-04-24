export interface WorkerRequestEnvelope<TRequest> {
  requestId: string;
  request: TRequest;
}

export interface WorkerSuccessEnvelope<TResponse> {
  requestId: string;
  ok: true;
  response: TResponse;
}

export interface WorkerEventEnvelope<TEvent> {
  requestId: string;
  ok: true;
  event: TEvent;
}

export interface WorkerFailureEnvelope {
  requestId: string;
  ok: false;
  error: string;
}

export type WorkerResponseEnvelope<TResponse, TEvent = never> =
  | WorkerSuccessEnvelope<TResponse>
  | WorkerEventEnvelope<TEvent>
  | WorkerFailureEnvelope;
