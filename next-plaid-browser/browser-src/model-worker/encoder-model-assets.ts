import { Context, Effect, Layer, Schema } from "effect";

import {
  type WorkerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import { FixtureTokenizer } from "./fixture-tokenizer.js";
import { parseOnnxConfig } from "./onnx-config.js";
import {
  EncoderModelAssetCache,
  type EncoderModelAssetCacheApi,
} from "./encoder-model-asset-cache.js";
import {
  EncoderInitEventSink,
  EncoderRuntimeConfig,
} from "./encoder-runtime-config.js";
import type { EncoderInitEvent, OnnxConfig } from "./types.js";

const decodeJsonString = Schema.decodeUnknownEffect(Schema.UnknownFromJsonString);

export interface ResolvedEncoderModelAssets {
  readonly modelBytes: Uint8Array;
  readonly tokenizer: FixtureTokenizer;
  readonly config: OnnxConfig;
  readonly persistentStorage: boolean;
}

export class EncoderModelAssets
  extends Context.Service<EncoderModelAssets, ResolvedEncoderModelAssets>()(
    "next-plaid-browser/EncoderModelAssets",
  )
{
  static readonly layer = Layer.effect(EncoderModelAssets)(
    makeEncoderModelAssets(),
  );
}

function loadAssetBytes(
  url: string,
  emitEvent: (event: EncoderInitEvent) => Effect.Effect<void>,
  modelAssetCache: EncoderModelAssetCacheApi,
): Effect.Effect<Uint8Array, WorkerRuntimeError> {
  return Effect.gen(function*() {
    yield* emitEvent({ stage: "fetch_start", url, expectedBytes: null });
    const bytes = yield* modelAssetCache.loadBytes(url);
    yield* emitEvent({
      stage: "fetch_complete",
      url,
      bytesReceived: bytes.byteLength,
    });
    return bytes;
  });
}

function loadAssetJson<T>(
  url: string,
  emitEvent: (event: EncoderInitEvent) => Effect.Effect<void>,
  modelAssetCache: EncoderModelAssetCacheApi,
  parse: (value: unknown) => Effect.Effect<T, WorkerRuntimeError>,
): Effect.Effect<T, WorkerRuntimeError> {
  return loadAssetBytes(url, emitEvent, modelAssetCache).pipe(
    Effect.map((bytes) => new TextDecoder().decode(bytes)),
    Effect.flatMap((text) =>
      decodeJsonString(text).pipe(
        Effect.mapError((error) =>
          workerRuntimeErrorFromUnknown(
            "encoder_model_assets.parse_asset_json",
            error,
            `failed to parse json asset ${url}`,
          ),
        ),
        Effect.flatMap(parse),
      ),
    ),
  );
}

function loadTokenizer(
  url: string,
  modelAssetCache: EncoderModelAssetCacheApi,
): Effect.Effect<FixtureTokenizer, WorkerRuntimeError> {
  return modelAssetCache.loadText(url).pipe(
    Effect.flatMap((text) =>
      decodeJsonString(text).pipe(
        Effect.mapError((error) =>
          workerRuntimeErrorFromUnknown(
            "encoder_model_assets.parse_tokenizer",
            error,
            `failed to parse tokenizer json ${url}`,
          ),
        ),
        Effect.flatMap((value) => FixtureTokenizer.fromJson(value)),
      ),
    ),
  );
}

function loadOnnxConfig(
  url: string,
  emitEvent: (event: EncoderInitEvent) => Effect.Effect<void>,
  modelAssetCache: EncoderModelAssetCacheApi,
): Effect.Effect<OnnxConfig, WorkerRuntimeError> {
  return loadAssetJson(url, emitEvent, modelAssetCache, (value) =>
    Effect.try({
      try: () => parseOnnxConfig(value),
      catch: (error) =>
        workerRuntimeErrorFromUnknown(
          "encoder_model_assets.decode_onnx_config",
          error,
          `failed to decode onnx config ${url}`,
        ),
    }),
  );
}

function makeEncoderModelAssets(): Effect.Effect<
  ResolvedEncoderModelAssets,
  WorkerRuntimeError,
  EncoderRuntimeConfig | EncoderInitEventSink | EncoderModelAssetCache
> {
  return Effect.gen(function*() {
    const { input } = yield* EncoderRuntimeConfig;
    const eventSink = yield* EncoderInitEventSink;
    const modelAssetCache = yield* EncoderModelAssetCache;

    const assets = yield* Effect.all({
      modelBytes: loadAssetBytes(
        input.modelUrl,
        eventSink.emit,
        modelAssetCache,
      ),
      tokenizer: loadTokenizer(input.tokenizerUrl, modelAssetCache),
      config: loadOnnxConfig(
        input.onnxConfigUrl,
        eventSink.emit,
        modelAssetCache,
      ),
      persistentStorage: modelAssetCache.persistentStorage(),
    });

    return EncoderModelAssets.of(assets);
  });
}
