import * as ort from "onnxruntime-web";

import type {
  EncodeTimingBreakdown,
  EncodedQuery,
  EncoderBackend,
  EncoderCapabilities,
  EncoderCreateInput,
  EncoderHealth,
  EncoderInitEvent,
} from "./types.js";
import { FixtureTokenizer } from "./fixture-tokenizer.js";
import { parseOnnxConfig } from "./onnx-config.js";

const MODEL_CACHE_NAME = "next-plaid-browser-model-worker-v1";

function nowMs(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function ensureFinite(values: Iterable<number>, label: string): void {
  for (const value of values) {
    if (!Number.isFinite(value)) {
      throw new Error(`${label} must not contain NaN or Infinity`);
    }
  }
}

async function loadAssetBytes(
  url: string,
  emitEvent: (event: EncoderInitEvent) => void,
): Promise<Uint8Array> {
  emitEvent({ stage: "fetch_start", url, expectedBytes: null });

  let response: Response | undefined;
  if (typeof caches !== "undefined") {
    const cache = await caches.open(MODEL_CACHE_NAME);
    response = await cache.match(url);
    if (!response) {
      response = await fetch(url);
      if (!response.ok) {
        throw new Error(`failed to fetch asset ${url}: ${response.status} ${response.statusText}`);
      }
      await cache.put(url, response.clone());
    }
  } else {
    response = await fetch(url);
    if (!response.ok) {
      throw new Error(`failed to fetch asset ${url}: ${response.status} ${response.statusText}`);
    }
  }

  const bytes = new Uint8Array(await response.arrayBuffer());
  emitEvent({ stage: "fetch_complete", url, bytesReceived: bytes.byteLength });
  return bytes;
}

async function loadAssetJson<T>(
  url: string,
  emitEvent: (event: EncoderInitEvent) => void,
  parse: (value: unknown) => T,
): Promise<T> {
  const bytes = await loadAssetBytes(url, emitEvent);
  const text = new TextDecoder().decode(bytes);
  return parse(JSON.parse(text) as unknown);
}

function configureOrt(): void {
  ort.env.wasm.wasmPaths = new URL("./ort/", self.location.href).toString();
  ort.env.wasm.proxy = false;
  ort.env.wasm.numThreads = 1;
}

function selectOutputTensor(results: ort.InferenceSession.ReturnType): ort.Tensor {
  const candidate = "output" in results ? results.output : Object.values(results)[0];
  if (!candidate || typeof candidate !== "object" || !("dims" in candidate)) {
    throw new Error("model inference did not produce a tensor output");
  }
  return candidate as ort.Tensor;
}

function toEmbeddingRows(data: Float32Array, rows: number, dim: number): number[][] {
  const embeddings: number[][] = [];
  for (let row = 0; row < rows; row += 1) {
    const start = row * dim;
    const values = Array.from(data.slice(start, start + dim));
    ensureFinite(values, "encoder output");
    embeddings.push(values);
  }
  return embeddings;
}

function buildTiming(total_ms: number, tokenize_ms: number, inference_ms: number): EncodeTimingBreakdown {
  return {
    total_ms,
    tokenize_ms,
    inference_ms,
  };
}

class WasmEncoderBackend implements EncoderBackend {
  readonly capabilities: EncoderCapabilities;

  readonly #session: ort.InferenceSession;
  readonly #config: EncoderCreateInput["encoder"] & { query_length: number; uses_token_type_ids: boolean; pad_token_id: number };
  readonly #tokenizer: FixtureTokenizer;
  #disposed = false;

  constructor(
    session: ort.InferenceSession,
    tokenizer: FixtureTokenizer,
    capabilities: EncoderCapabilities,
    config: { query_length: number; uses_token_type_ids: boolean; pad_token_id: number } & EncoderCreateInput["encoder"],
  ) {
    this.#session = session;
    this.#tokenizer = tokenizer;
    this.capabilities = capabilities;
    this.#config = config;
  }

  async encode(text: string): Promise<EncodedQuery> {
    if (this.#disposed) {
      throw new Error("encoder backend is disposed");
    }

    const totalStartedAt = nowMs();
    const tokenizeStartedAt = nowMs();
    const tokenized = this.#tokenizer.encodeQuery(text, {
      query_prefix: "",
      document_prefix: "",
      query_length: this.#config.query_length,
      document_length: this.#config.query_length,
      do_query_expansion: false,
      embedding_dim: this.#config.embedding_dim,
      uses_token_type_ids: this.#config.uses_token_type_ids,
      mask_token_id: this.#config.pad_token_id,
      pad_token_id: this.#config.pad_token_id,
      skiplist_words: [],
      do_lower_case: false,
    });
    const tokenize_ms = nowMs() - tokenizeStartedAt;

    const feeds: ort.InferenceSession.FeedsType = {
      input_ids: new ort.Tensor("int64", tokenized.inputIds, [1, this.#config.query_length]),
      attention_mask: new ort.Tensor("int64", tokenized.attentionMask, [1, this.#config.query_length]),
    };

    const inferenceStartedAt = nowMs();
    const results = await this.#session.run(feeds);
    const inference_ms = nowMs() - inferenceStartedAt;
    const output = selectOutputTensor(results);
    if (output.type !== "float32") {
      throw new Error(`expected float32 encoder output, got ${output.type}`);
    }
    const data = await output.getData();
    if (!(data instanceof Float32Array)) {
      throw new Error("expected float32 encoder output buffer");
    }
    if (output.dims.length !== 3) {
      throw new Error(`expected 3D encoder output, got dims ${output.dims.join("x")}`);
    }
    const [, rows, dim] = output.dims;
    if (rows !== this.#config.query_length || dim !== this.#config.embedding_dim) {
      throw new Error(
        `unexpected encoder output shape ${output.dims.join("x")} expected 1x${this.#config.query_length}x${this.#config.embedding_dim}`,
      );
    }

    return {
      payload: {
        embeddings: toEmbeddingRows(data, rows, dim),
        encoder: {
          encoder_id: this.#config.encoder_id,
          encoder_build: this.#config.encoder_build,
          embedding_dim: this.#config.embedding_dim,
          normalized: this.#config.normalized,
        },
        dtype: "f32_le",
        layout: "padded_query_length",
      },
      timing: buildTiming(nowMs() - totalStartedAt, tokenize_ms, inference_ms),
      input_ids: tokenized.inputIdValues,
      attention_mask: tokenized.attentionMaskValues,
    };
  }

  async health(): Promise<EncoderHealth> {
    return this.#disposed ? "degraded" : "ok";
  }

  async dispose(): Promise<void> {
    if (!this.#disposed) {
      await this.#session.release();
      this.#disposed = true;
    }
  }
}

export async function createWasmEncoderBackend(
  input: EncoderCreateInput,
  emitEvent: (event: EncoderInitEvent) => void,
): Promise<EncoderBackend> {
  configureOrt();

  const [modelBytes, tokenizer, config] = await Promise.all([
    loadAssetBytes(input.modelUrl, emitEvent),
    FixtureTokenizer.load(input.tokenizerUrl),
    loadAssetJson(input.onnxConfigUrl, emitEvent, parseOnnxConfig),
  ]);

  const persistentStorage =
    typeof navigator !== "undefined" &&
    "storage" in navigator &&
    typeof navigator.storage?.persisted === "function"
      ? await navigator.storage.persisted()
      : false;

  emitEvent({ stage: "session_create_start" });
  const sessionStartedAt = nowMs();
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  emitEvent({ stage: "session_create_complete", durationMs: nowMs() - sessionStartedAt });

  emitEvent({ stage: "warmup_start" });
  const warmupStartedAt = nowMs();
  const warmupFeeds: ort.InferenceSession.FeedsType = {
    input_ids: new ort.Tensor("int64", new BigInt64Array(config.query_length), [1, config.query_length]),
    attention_mask: new ort.Tensor(
      "int64",
      BigInt64Array.from({ length: config.query_length }, () => 1n),
      [1, config.query_length],
    ),
  };
  await session.run(warmupFeeds);
  emitEvent({ stage: "warmup_complete", durationMs: nowMs() - warmupStartedAt });

  const capabilities: EncoderCapabilities = {
    backend: "wasm",
    threaded: false,
    persistentStorage,
    encoderId: input.encoder.encoder_id,
    encoderBuild: input.encoder.encoder_build,
    embeddingDim: config.embedding_dim,
    queryLength: config.query_length,
    doQueryExpansion: config.do_query_expansion,
    normalized: input.encoder.normalized,
  };
  emitEvent({ stage: "ready", capabilities });

  return new WasmEncoderBackend(session, tokenizer, capabilities, {
    ...input.encoder,
    query_length: config.query_length,
    uses_token_type_ids: config.uses_token_type_ids,
    pad_token_id: config.pad_token_id,
  });
}
