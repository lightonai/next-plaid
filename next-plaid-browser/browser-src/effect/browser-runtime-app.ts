import * as BrowserWorker from "@effect/platform-browser/BrowserWorker";
import {
  Layer,
  ManagedRuntime,
} from "effect";
import type * as Worker from "effect/unstable/workers/Worker";

import type {
  EncoderClientError,
  SearchClientError,
} from "./client-errors.js";
import { BrowserSearchRuntime } from "./browser-search-runtime.js";
import {
  EncoderWorkerClient,
  type EncoderWorkerClientOptions,
} from "./encoder-worker-client.js";
import {
  SearchWorkerClient,
  type SearchWorkerClientOptions,
} from "./search-worker-client.js";

export type BrowserRuntimeAppServices =
  | SearchWorkerClient
  | EncoderWorkerClient
  | BrowserSearchRuntime;

export type BrowserRuntimeAppError =
  | SearchClientError
  | EncoderClientError;

export interface BrowserWorkerFactories {
  readonly searchWorker: () => globalThis.Worker;
  readonly encoderWorker: () => globalThis.Worker;
}

export interface BrowserRuntimeWorkerLayers {
  readonly searchWorkerLayer: Layer.Layer<
    Worker.WorkerPlatform | Worker.Spawner,
    never
  >;
  readonly encoderWorkerLayer: Layer.Layer<
    Worker.WorkerPlatform | Worker.Spawner,
    never
  >;
}

export interface BrowserRuntimeCompositionOptions extends BrowserRuntimeWorkerLayers {
  readonly searchClientOptions?: SearchWorkerClientOptions | undefined;
  readonly encoderClientOptions?: EncoderWorkerClientOptions | undefined;
}

export function makeBrowserRuntimeWorkerLayers(
  factories: BrowserWorkerFactories,
): BrowserRuntimeWorkerLayers {
  return {
    searchWorkerLayer: BrowserWorker.layer(factories.searchWorker),
    encoderWorkerLayer: BrowserWorker.layer(factories.encoderWorker),
  };
}

export function makeBrowserWorkerClientLayer(
  options: BrowserRuntimeCompositionOptions,
): Layer.Layer<
  SearchWorkerClient | EncoderWorkerClient,
  BrowserRuntimeAppError
> {
  const searchClientLayer = SearchWorkerClient.layer(options.searchClientOptions).pipe(
    Layer.provide(options.searchWorkerLayer),
  );
  const encoderClientLayer = EncoderWorkerClient.layer(options.encoderClientOptions).pipe(
    Layer.provide(options.encoderWorkerLayer),
  );

  return Layer.mergeAll(searchClientLayer, encoderClientLayer);
}

export function makeBrowserSearchRuntimeLayer(
  options: BrowserRuntimeCompositionOptions,
): Layer.Layer<
  BrowserRuntimeAppServices,
  BrowserRuntimeAppError
> {
  const workerClientLayer = makeBrowserWorkerClientLayer(options);
  const runtimeLayer = BrowserSearchRuntime.layer().pipe(
    Layer.provide(workerClientLayer),
  );

  return Layer.mergeAll(workerClientLayer, runtimeLayer);
}

export function makeBrowserSearchRuntimeManagedRuntime(
  options: BrowserRuntimeCompositionOptions,
): ManagedRuntime.ManagedRuntime<
  BrowserRuntimeAppServices,
  BrowserRuntimeAppError
> {
  return ManagedRuntime.make(makeBrowserSearchRuntimeLayer(options));
}

export function makeBrowserSearchRuntimeManagedRuntimeFromFactories(
  options: BrowserWorkerFactories & {
    readonly searchClientOptions?: SearchWorkerClientOptions | undefined;
    readonly encoderClientOptions?: EncoderWorkerClientOptions | undefined;
  },
): ManagedRuntime.ManagedRuntime<
  BrowserRuntimeAppServices,
  BrowserRuntimeAppError
> {
  return makeBrowserSearchRuntimeManagedRuntime({
    ...makeBrowserRuntimeWorkerLayers(options),
    searchClientOptions: options.searchClientOptions,
    encoderClientOptions: options.encoderClientOptions,
  });
}
