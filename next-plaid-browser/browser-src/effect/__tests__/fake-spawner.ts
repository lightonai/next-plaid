export interface CapturedRequest<TRequest = unknown> {
  readonly requestId: string;
  readonly request: TRequest;
}

export type OutboundFrame =
  | readonly [0, CapturedRequest]
  | readonly [1];

export interface FakeSpawner {
  readonly spawn: (id: number) => FakeWorkerPort;
  readonly port: FakeWorkerPort;
  readonly dispatchReady: () => void;
  readonly dispatchEnvelope: (envelope: unknown) => void;
  readonly dispatchError: (
    options?: { readonly message?: string; readonly error?: unknown } | undefined,
  ) => void;
  readonly dispatchMessageError: (data?: unknown) => void;
  readonly capturedOutbound: () => ReadonlyArray<OutboundFrame>;
  readonly capturedRequests: <TRequest = unknown>() => ReadonlyArray<CapturedRequest<TRequest>>;
  readonly clearOutbound: () => void;
  readonly spawnCount: () => number;
  readonly spawnedIds: () => ReadonlyArray<number>;
  readonly isStarted: () => boolean;
}

function isCapturedRequest(value: unknown): value is CapturedRequest {
  return (
    typeof value === "object" &&
    value !== null &&
    "requestId" in value &&
    typeof value.requestId === "string" &&
    "request" in value
  );
}

export class FakeWorkerPort extends EventTarget {
  private readonly outbound: Array<OutboundFrame> = [];
  private started = false;
  onmessage: ((this: MessagePort, ev: MessageEvent) => unknown) | null = null;
  onmessageerror: ((this: MessagePort, ev: MessageEvent) => unknown) | null = null;

  postMessage(message: unknown, _transfers?: unknown): void {
    this.outbound.push(message as OutboundFrame);
  }

  close(): void {
    this.started = false;
  }

  start(): void {
    this.started = true;
  }

  dispatchReady(): void {
    const event = new MessageEvent("message", { data: [0] as const });
    this.onmessage?.call(this as unknown as MessagePort, event);
    this.dispatchEvent(event);
  }

  dispatchEnvelope(envelope: unknown): void {
    const event = new MessageEvent("message", { data: [1, envelope] as const });
    this.onmessage?.call(this as unknown as MessagePort, event);
    this.dispatchEvent(event);
  }

  dispatchError(
    options?: { readonly message?: string; readonly error?: unknown } | undefined,
  ): void {
    const event = new Event("error");
    if (options?.message !== undefined) {
      Object.defineProperty(event, "message", {
        configurable: true,
        value: options.message,
      });
    }
    if (options?.error !== undefined) {
      Object.defineProperty(event, "error", {
        configurable: true,
        value: options.error,
      });
    }
    this.dispatchEvent(event);
  }

  dispatchMessageError(data?: unknown): void {
    const event = new MessageEvent("messageerror", { data });
    this.onmessageerror?.call(this as unknown as MessagePort, event);
    this.dispatchEvent(event);
  }

  capturedOutbound(): ReadonlyArray<OutboundFrame> {
    return [...this.outbound];
  }

  capturedRequests<TRequest = unknown>(): ReadonlyArray<CapturedRequest<TRequest>> {
    const requests: Array<CapturedRequest<TRequest>> = [];
    for (const frame of this.outbound) {
      if (frame[0] !== 0) {
        continue;
      }
      if (isCapturedRequest(frame[1])) {
        requests.push(frame[1] as CapturedRequest<TRequest>);
      }
    }
    return requests;
  }

  clearOutbound(): void {
    this.outbound.length = 0;
  }

  isStarted(): boolean {
    return this.started;
  }
}

export function makeFakeSpawner(): FakeSpawner {
  const port = new FakeWorkerPort();
  const spawnedIds: Array<number> = [];

  return {
    spawn(id) {
      spawnedIds.push(id);
      return port;
    },
    port,
    dispatchReady: () => port.dispatchReady(),
    dispatchEnvelope: (envelope) => port.dispatchEnvelope(envelope),
    dispatchError: (options) => port.dispatchError(options),
    dispatchMessageError: (data) => port.dispatchMessageError(data),
    capturedOutbound: () => port.capturedOutbound(),
    capturedRequests: () => port.capturedRequests(),
    clearOutbound: () => port.clearOutbound(),
    spawnCount: () => spawnedIds.length,
    spawnedIds: () => [...spawnedIds],
    isStarted: () => port.isStarted(),
  };
}
