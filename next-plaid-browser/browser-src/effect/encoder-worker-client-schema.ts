import { Schema } from "effect";

import type {
  EncoderCapabilities,
  EncoderCreateInput,
} from "../model-worker/types.js";
import { EncoderIdentitySchema } from "../shared/search-contract-schema.js";

const EncoderPreferenceSchema = Schema.NullOr(
  Schema.Union([Schema.Literal("wasm"), Schema.Literal("auto")]),
);

const EncoderInitBindingKeySchema = Schema.Struct({
  encoder: EncoderIdentitySchema,
  modelUrl: Schema.String,
  onnxConfigUrl: Schema.String,
  tokenizerUrl: Schema.String,
  prefer: EncoderPreferenceSchema,
});

const EncoderInitBindingKeyJsonSchema = Schema.fromJsonString(
  EncoderInitBindingKeySchema,
);

export const EncoderQueryCacheKeySchema = Schema.Struct({
  encoderId: Schema.String,
  encoderBuild: Schema.String,
  text: Schema.String,
});

const EncoderQueryCacheKeyJsonSchema = Schema.fromJsonString(
  EncoderQueryCacheKeySchema,
);

export type EncoderQueryCacheKey = Schema.Schema.Type<
  typeof EncoderQueryCacheKeySchema
>;

export const isEncoderQueryCacheKey = Schema.is(EncoderQueryCacheKeySchema);

function normalizeEncoderInitBindingKey(
  input: EncoderCreateInput,
): Schema.Schema.Type<typeof EncoderInitBindingKeySchema> {
  return {
    encoder: input.encoder,
    modelUrl: input.modelUrl,
    onnxConfigUrl: input.onnxConfigUrl,
    tokenizerUrl: input.tokenizerUrl,
    prefer: input.prefer ?? null,
  };
}

function normalizeEncoderQueryCacheKey(
  capabilities: EncoderCapabilities,
  text: string,
): EncoderQueryCacheKey {
  return {
    encoderId: capabilities.encoderId,
    encoderBuild: capabilities.encoderBuild,
    text,
  };
}

export const encodeEncoderInitBindingKey = (
  input: EncoderCreateInput,
) =>
  Schema.encodeEffect(EncoderInitBindingKeyJsonSchema)(
    normalizeEncoderInitBindingKey(input),
  );

export const encodeEncoderQueryCacheKey = (
  capabilities: EncoderCapabilities,
  text: string,
) =>
  Schema.encodeEffect(EncoderQueryCacheKeyJsonSchema)(
    normalizeEncoderQueryCacheKey(capabilities, text),
  );

export const decodeEncoderQueryCacheKey = Schema.decodeUnknownEffect(
  EncoderQueryCacheKeyJsonSchema,
);
