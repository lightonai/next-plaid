import { Effect, Schema } from "effect";

import type {
  EncodeResponse,
  EncoderCreateInput,
  EncoderWorkerRequest,
} from "./types.js";

export const EncoderIdentitySchema = Schema.Struct({
  encoder_id: Schema.String,
  encoder_build: Schema.String,
  embedding_dim: Schema.Number,
  normalized: Schema.Boolean,
});

export const EncoderCapabilitiesSchema = Schema.Struct({
  backend: Schema.Literal("wasm"),
  threaded: Schema.Boolean,
  persistentStorage: Schema.Boolean,
  encoderId: Schema.String,
  encoderBuild: Schema.String,
  embeddingDim: Schema.Number,
  queryLength: Schema.Number,
  doQueryExpansion: Schema.Boolean,
  normalized: Schema.Boolean,
});

export const EncodeTimingBreakdownSchema = Schema.Struct({
  total_ms: Schema.Number,
  tokenize_ms: Schema.Number,
  inference_ms: Schema.Number,
});

export const EncodedQuerySchema = Schema.Struct({
  payload: Schema.Struct({
    embeddings: Schema.Array(Schema.Array(Schema.Number)),
    encoder: EncoderIdentitySchema,
    dtype: Schema.Literal("f32_le"),
    layout: Schema.Union([
      Schema.Literal("ragged"),
      Schema.Literal("padded_query_length"),
    ]),
  }),
  timing: EncodeTimingBreakdownSchema,
  input_ids: Schema.Array(Schema.Number),
  attention_mask: Schema.Array(Schema.Number),
});

export const EncoderCreateInputSchema = Schema.Struct({
  encoder: EncoderIdentitySchema,
  modelUrl: Schema.String,
  onnxConfigUrl: Schema.String,
  tokenizerUrl: Schema.String,
  prefer: Schema.optional(
    Schema.Union([Schema.Literal("wasm"), Schema.Literal("auto")]),
  ),
});

export const EncoderInitResponseSchema = Schema.Struct({
  type: Schema.Literal("encoder_ready"),
  state: Schema.Literal("ready"),
  capabilities: EncoderCapabilitiesSchema,
});

export const EncodeResponseSchema = Schema.Struct({
  type: Schema.Literal("encoded_query"),
  encoded: EncodedQuerySchema,
});

export const EncoderInitEventSchema = Schema.Union([
  Schema.Struct({
    stage: Schema.Literal("fetch_start"),
    url: Schema.String,
    expectedBytes: Schema.NullOr(Schema.Number),
  }),
  Schema.Struct({
    stage: Schema.Literal("fetch_complete"),
    url: Schema.String,
    bytesReceived: Schema.Number,
  }),
  Schema.Struct({
    stage: Schema.Literal("session_create_start"),
  }),
  Schema.Struct({
    stage: Schema.Literal("session_create_complete"),
    durationMs: Schema.Number,
  }),
  Schema.Struct({
    stage: Schema.Literal("warmup_start"),
  }),
  Schema.Struct({
    stage: Schema.Literal("warmup_complete"),
    durationMs: Schema.Number,
  }),
  Schema.Struct({
    stage: Schema.Literal("ready"),
    capabilities: EncoderCapabilitiesSchema,
  }),
]);

export const EncoderWorkerRequestSchema = Schema.Union([
  Schema.Struct({
    type: Schema.Literal("init"),
    payload: EncoderCreateInputSchema,
  }),
  Schema.Struct({
    type: Schema.Literal("health"),
  }),
  Schema.Struct({
    type: Schema.Literal("encode"),
    payload: Schema.Struct({
      text: Schema.String,
    }),
  }),
  Schema.Struct({
    type: Schema.Literal("dispose"),
  }),
]);

function normalizeEncoderCreateInput(
  input: Schema.Schema.Type<typeof EncoderCreateInputSchema>,
): EncoderCreateInput {
  if (input.prefer === undefined) {
    return {
      encoder: input.encoder,
      modelUrl: input.modelUrl,
      onnxConfigUrl: input.onnxConfigUrl,
      tokenizerUrl: input.tokenizerUrl,
    };
  }

  return {
    encoder: input.encoder,
    modelUrl: input.modelUrl,
    onnxConfigUrl: input.onnxConfigUrl,
    tokenizerUrl: input.tokenizerUrl,
    prefer: input.prefer,
  };
}

function normalizeEncodeResponse(
  response: Schema.Schema.Type<typeof EncodeResponseSchema>,
): EncodeResponse {
  return {
    type: response.type,
    encoded: {
      payload: {
        embeddings: response.encoded.payload.embeddings.map((row) => [...row]),
        encoder: response.encoded.payload.encoder,
        dtype: response.encoded.payload.dtype,
        layout: response.encoded.payload.layout,
      },
      timing: {
        total_ms: response.encoded.timing.total_ms,
        tokenize_ms: response.encoded.timing.tokenize_ms,
        inference_ms: response.encoded.timing.inference_ms,
      },
      input_ids: [...response.encoded.input_ids],
      attention_mask: [...response.encoded.attention_mask],
    },
  };
}

function normalizeEncoderWorkerRequest(
  request: Schema.Schema.Type<typeof EncoderWorkerRequestSchema>,
): EncoderWorkerRequest {
  if (request.type !== "init") {
    return request;
  }

  return {
    type: "init",
    payload: normalizeEncoderCreateInput(request.payload),
  };
}

export const decodeEncoderWorkerRequest = (
  value: unknown,
): Effect.Effect<EncoderWorkerRequest, unknown> =>
  Schema.decodeUnknownEffect(EncoderWorkerRequestSchema)(value).pipe(
    Effect.map(normalizeEncoderWorkerRequest),
  );

export const decodeEncoderInitResponseSchema = Schema.decodeUnknownEffect(
  EncoderInitResponseSchema,
);

export const decodeEncodeResponseSchema = (
  value: unknown,
): Effect.Effect<EncodeResponse, unknown> =>
  Schema.decodeUnknownEffect(EncodeResponseSchema)(value).pipe(
    Effect.map(normalizeEncodeResponse),
  );

export const decodeEncoderInitEventSchema = Schema.decodeUnknownEffect(
  EncoderInitEventSchema,
);
