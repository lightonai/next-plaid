import { Schema } from "effect";

export const ModelAssetStoreKindSchema = Schema.Union([
  Schema.Literal("opfs"),
  Schema.Literal("indexeddb"),
  Schema.Literal("transient"),
]);

export const DurableModelAssetStoreKindSchema = Schema.Union([
  Schema.Literal("opfs"),
  Schema.Literal("indexeddb"),
]);

export const ModelAssetKindSchema = Schema.Union([
  Schema.Literal("model"),
  Schema.Literal("tokenizer"),
  Schema.Literal("onnxConfig"),
]);
