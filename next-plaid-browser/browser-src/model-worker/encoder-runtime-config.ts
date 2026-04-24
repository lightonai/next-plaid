import { Context, Effect, Layer } from "effect";

import type { EncoderCreateInput, EncoderInitEvent } from "./types.js";

interface EncoderRuntimeConfigApi {
  readonly input: EncoderCreateInput;
}

export class EncoderRuntimeConfig
  extends Context.Service<EncoderRuntimeConfig, EncoderRuntimeConfigApi>()(
    "next-plaid-browser/EncoderRuntimeConfig",
  )
{
  static layer = (
    input: EncoderCreateInput,
  ): Layer.Layer<EncoderRuntimeConfig> =>
    Layer.succeed(EncoderRuntimeConfig, EncoderRuntimeConfig.of({ input }));
}

interface EncoderInitEventSinkApi {
  readonly emit: (event: EncoderInitEvent) => Effect.Effect<void>;
}

export class EncoderInitEventSink
  extends Context.Service<EncoderInitEventSink, EncoderInitEventSinkApi>()(
    "next-plaid-browser/EncoderInitEventSink",
  )
{
  static layer = (
    emit: (event: EncoderInitEvent) => Effect.Effect<void>,
  ): Layer.Layer<EncoderInitEventSink> =>
    Layer.succeed(EncoderInitEventSink, EncoderInitEventSink.of({ emit }));
}
