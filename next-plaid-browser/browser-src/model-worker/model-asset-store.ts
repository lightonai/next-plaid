import * as IndexedDb from "@effect/platform-browser/IndexedDb";
import * as IndexedDbDatabase from "@effect/platform-browser/IndexedDbDatabase";
import * as IndexedDbTable from "@effect/platform-browser/IndexedDbTable";
import * as IndexedDbVersion from "@effect/platform-browser/IndexedDbVersion";
import {
  Cause,
  Clock,
  Context,
  Effect,
  Layer,
  Ref,
  Schema,
} from "effect";
import * as Reactivity from "effect/unstable/reactivity/Reactivity";

import {
  WorkerRuntimeError,
  workerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import {
  type DurableModelAssetStoreKind,
  MODEL_ASSET_DB_NAME,
  MODEL_ASSET_KINDS,
  MODEL_ASSET_OPFS_ROOT_DIR,
  type ModelAssetKind,
  type ModelAssetPackage,
  type ModelAssetPackageKey,
  type ModelAssetStoreKind,
  modelAssetBytes,
  supportsIndexedDb,
  supportsOpfs,
} from "./model-asset-types.js";
import {
  DurableModelAssetStoreKindSchema,
  ModelAssetKindSchema,
} from "./model-asset-store-schema.js";

type ModelAssetRecordState = "staging" | "ready";

interface ModelAssetPackageRecord {
  readonly packageId: string;
  readonly state: ModelAssetRecordState;
  readonly backend: DurableModelAssetStoreKind;
  readonly encoderId: string;
  readonly encoderBuild: string;
  readonly embeddingDim: number;
  readonly normalized: boolean;
  readonly modelUrl: string;
  readonly tokenizerUrl: string;
  readonly onnxConfigUrl: string;
  readonly payloadRef: string;
  readonly createdAtMs: number;
  readonly lastAccessedAtMs: number;
}

interface ModelAssetStoreApi {
  readonly loadPackage: (
    key: ModelAssetPackageKey,
  ) => Effect.Effect<ModelAssetPackage | null, WorkerRuntimeError>;
  readonly storePackage: (
    pkg: ModelAssetPackage,
  ) => Effect.Effect<void, WorkerRuntimeError>;
  readonly removePackage: (
    packageId: string,
  ) => Effect.Effect<void, WorkerRuntimeError>;
  readonly persistentStorage: () => Effect.Effect<boolean, WorkerRuntimeError>;
  readonly kind: () => Effect.Effect<ModelAssetStoreKind>;
}

const ModelAssetRecordStateSchema = Schema.Union([
  Schema.Literal("staging"),
  Schema.Literal("ready"),
]);
const ModelAssetPackageRecordSchema = Schema.Struct({
  packageId: Schema.String,
  state: ModelAssetRecordStateSchema,
  backend: DurableModelAssetStoreKindSchema,
  encoderId: Schema.String,
  encoderBuild: Schema.String,
  embeddingDim: Schema.Number,
  normalized: Schema.Boolean,
  modelUrl: Schema.String,
  tokenizerUrl: Schema.String,
  onnxConfigUrl: Schema.String,
  payloadRef: Schema.String,
  createdAtMs: Schema.Number,
  lastAccessedAtMs: Schema.Number,
});

const ModelAssetBlobRecordSchema = Schema.Struct({
  packageId: Schema.String,
  assetKind: ModelAssetKindSchema,
  bytes: Schema.Uint8Array,
});

const ModelAssetPackageTable = IndexedDbTable.make({
  name: "packages",
  schema: ModelAssetPackageRecordSchema,
  keyPath: "packageId",
});

const ModelAssetBlobTable = IndexedDbTable.make({
  name: "asset_blobs",
  schema: ModelAssetBlobRecordSchema,
  keyPath: ["packageId", "assetKind"],
  indexes: {
    byPackage: "packageId",
  },
});

const ModelAssetCatalogVersion = IndexedDbVersion.make(
  ModelAssetPackageTable,
  ModelAssetBlobTable,
);

const ModelAssetCatalogDatabase = IndexedDbDatabase.make(
  ModelAssetCatalogVersion,
  (tx) =>
    Effect.gen(function*() {
      yield* tx.createObjectStore("packages");
      yield* tx.createObjectStore("asset_blobs");
      yield* tx.createIndex("asset_blobs", "byPackage");
    }),
);

type ModelAssetArea = "packages" | "staging";

interface ModelAssetCatalogApi {
  readonly getRecord: (
    packageId: string,
  ) => Effect.Effect<ModelAssetPackageRecord | null, WorkerRuntimeError>;
  readonly putRecord: (
    record: ModelAssetPackageRecord,
  ) => Effect.Effect<void, WorkerRuntimeError>;
  readonly touchRecord: (
    packageId: string,
    lastAccessedAtMs: number,
  ) => Effect.Effect<void, WorkerRuntimeError>;
  readonly removePackage: (
    packageId: string,
  ) => Effect.Effect<void, WorkerRuntimeError>;
  readonly loadIndexedDbPackage: (
    key: ModelAssetPackageKey,
  ) => Effect.Effect<ModelAssetPackage | null, WorkerRuntimeError>;
  readonly commitIndexedDbPackage: (
    record: ModelAssetPackageRecord,
    pkg: ModelAssetPackage,
  ) => Effect.Effect<void, WorkerRuntimeError>;
}

interface ModelAssetOpfsStoreApi {
  readonly loadPackage: (
    area: ModelAssetArea,
    ref: string,
    key: ModelAssetPackageKey,
  ) => Effect.Effect<ModelAssetPackage | null, WorkerRuntimeError>;
  readonly writePackage: (
    area: ModelAssetArea,
    ref: string,
    pkg: ModelAssetPackage,
  ) => Effect.Effect<void, WorkerRuntimeError>;
  readonly removePackage: (
    area: ModelAssetArea,
    ref: string,
  ) => Effect.Effect<void, WorkerRuntimeError>;
}

export class ModelAssetStore
  extends Context.Service<ModelAssetStore, ModelAssetStoreApi>()(
    "next-plaid-browser/ModelAssetStore",
  )
{
  static get layerOpfs(): Layer.Layer<ModelAssetStore, WorkerRuntimeError> {
    return Layer.effect(ModelAssetStore)(
      makeOpfsModelAssetStore(),
    ).pipe(
      Layer.provide(Layer.mergeAll(ModelAssetCatalog.layer, ModelAssetOpfsStore.layer)),
    );
  }

  static get layerIndexedDb(): Layer.Layer<ModelAssetStore, WorkerRuntimeError> {
    return Layer.effect(ModelAssetStore)(
      makeIndexedDbModelAssetStore(),
    ).pipe(
      Layer.provide(ModelAssetCatalog.layer),
    );
  }

  static get layerTransient(): Layer.Layer<ModelAssetStore> {
    return Layer.effect(ModelAssetStore)(
      makeTransientModelAssetStore(),
    );
  }

  static get layerAuto(): Layer.Layer<ModelAssetStore> {
    return ModelAssetStore.layerOpfs.pipe(
      Layer.catchTag(
        "WorkerRuntimeError",
        () =>
          ModelAssetStore.layerIndexedDb.pipe(
            Layer.catchTag(
              "WorkerRuntimeError",
              () => ModelAssetStore.layerTransient,
            ),
          ),
      ),
    );
  }
}

class ModelAssetCatalog
  extends Context.Service<ModelAssetCatalog, ModelAssetCatalogApi>()(
    "next-plaid-browser/ModelAssetCatalog",
  )
{
  static readonly layer = Layer.effect(ModelAssetCatalog)(
    makeModelAssetCatalog(),
  ).pipe(
    Layer.provide(
      ModelAssetCatalogDatabase.layer(MODEL_ASSET_DB_NAME).pipe(
        Layer.provide(
          Layer.mergeAll(globalIndexedDbLayer(), Reactivity.layer),
        ),
      ),
    ),
    Layer.catchCause((cause) =>
      Layer.effect(ModelAssetCatalog)(
        Effect.fail(
          (() => {
            const error = Cause.squash(cause);
            return error instanceof WorkerRuntimeError
              ? error
              : workerRuntimeErrorFromUnknown(
                  "model_asset_catalog.layer",
                  error,
                  "failed to initialize model asset catalog database",
                );
          })(),
        ),
      ),
    ),
  );
}

class ModelAssetOpfsStore
  extends Context.Service<ModelAssetOpfsStore, ModelAssetOpfsStoreApi>()(
    "next-plaid-browser/ModelAssetOpfsStore",
  )
{
  static readonly layer = Layer.effect(ModelAssetOpfsStore)(
    makeModelAssetOpfsStore(),
  );
}

function modelAssetRecord(
  key: ModelAssetPackageKey,
  options: {
    readonly state: ModelAssetRecordState;
    readonly backend: DurableModelAssetStoreKind;
    readonly payloadRef: string;
    readonly createdAtMs: number;
    readonly lastAccessedAtMs: number;
  },
): ModelAssetPackageRecord {
  return {
    packageId: key.packageId,
    state: options.state,
    backend: options.backend,
    encoderId: key.encoderId,
    encoderBuild: key.encoderBuild,
    embeddingDim: key.embeddingDim,
    normalized: key.normalized,
    modelUrl: key.modelUrl,
    tokenizerUrl: key.tokenizerUrl,
    onnxConfigUrl: key.onnxConfigUrl,
    payloadRef: options.payloadRef,
    createdAtMs: options.createdAtMs,
    lastAccessedAtMs: options.lastAccessedAtMs,
  };
}

function packageFromBytes(
  key: ModelAssetPackageKey,
  bytesByKind: ReadonlyMap<ModelAssetKind, Uint8Array>,
): ModelAssetPackage | null {
  const modelBytes = bytesByKind.get("model");
  const tokenizerBytes = bytesByKind.get("tokenizer");
  const onnxConfigBytes = bytesByKind.get("onnxConfig");

  if (
    modelBytes === undefined ||
    tokenizerBytes === undefined ||
    onnxConfigBytes === undefined
  ) {
    return null;
  }

  return {
    key,
    modelBytes,
    tokenizerBytes,
    onnxConfigBytes,
  };
}

function globalIndexedDbLayer(): Layer.Layer<
  IndexedDb.IndexedDb,
  WorkerRuntimeError
> {
  return Layer.effect(IndexedDb.IndexedDb)(
    Effect.gen(function*() {
      if (!supportsIndexedDb()) {
        return yield* workerRuntimeError({
          operation: "model_asset_store.indexeddb_layer",
          message: "indexedDB is unavailable in this worker",
        });
      }

      return IndexedDb.make({
        indexedDB: globalThis.indexedDB,
        IDBKeyRange: globalThis.IDBKeyRange,
      });
    }),
  );
}

interface TransactionCallbackHandlers<A> {
  readonly succeed: (value: A) => void;
  readonly fail: (error: unknown) => void;
  readonly registerRequest: <T>(request: IDBRequest<T>) => IDBRequest<T>;
}

function abortTransaction(
  transaction: IDBTransaction,
): Effect.Effect<void> {
  return Effect.try({
    try: () => {
      transaction.abort();
    },
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "model_asset_catalog.abort_transaction",
        error,
        "failed to abort model asset transaction",
      ),
  }).pipe(Effect.ignore);
}

function runTransactionCallback<A>(options: {
  readonly transaction: IDBTransaction;
  readonly operation: string;
  readonly failureMessage: string;
  readonly register: (handlers: TransactionCallbackHandlers<A>) => void;
}): Effect.Effect<A, WorkerRuntimeError> {
  return Effect.callback<A, WorkerRuntimeError>((resume) => {
    let settled = false;
    const requests: Array<IDBRequest<unknown>> = [];

    const detach = () => {
      options.transaction.onabort = null;
      options.transaction.onerror = null;
      options.transaction.oncomplete = null;
      for (const request of requests) {
        request.onerror = null;
        request.onsuccess = null;
      }
    };

    const settle = (effect: Effect.Effect<A, WorkerRuntimeError>) => {
      if (settled) {
        return;
      }
      settled = true;
      detach();
      resume(effect);
    };

    const fail = (error: unknown) =>
      settle(
        Effect.fail(
          workerRuntimeErrorFromUnknown(
            options.operation,
            error,
            options.failureMessage,
          ),
        ),
      );

    options.transaction.onabort = (event) => {
      fail(options.transaction.error ?? event);
    };
    options.transaction.onerror = (event) => {
      fail(options.transaction.error ?? event);
    };

    options.register({
      succeed: (value) => {
        settle(Effect.succeed(value));
      },
      fail,
      registerRequest: <T>(request: IDBRequest<T>) => {
        requests.push(request as IDBRequest<unknown>);
        return request;
      },
    });

    return Effect.sync(detach).pipe(
      Effect.flatMap(() => abortTransaction(options.transaction)),
      Effect.ignore,
    );
  });
}

function removeOpfsPackageIfAvailable(
  area: ModelAssetArea,
  ref: string,
): Effect.Effect<void> {
  if (!supportsOpfs()) {
    return Effect.void;
  }

  return Effect.tryPromise({
    try: async () => {
      try {
        const root = await navigator.storage.getDirectory();
        const modelAssetRoot = await root.getDirectoryHandle(
          MODEL_ASSET_OPFS_ROOT_DIR,
          {
            create: false,
          },
        );
        const areaDirectory = await modelAssetRoot.getDirectoryHandle(area, {
          create: false,
        });
        await areaDirectory.removeEntry(ref, { recursive: true });
      } catch (error) {
        if (
          error instanceof DOMException &&
          error.name === "NotFoundError"
        ) {
          return;
        }
        throw error;
      }
    },
    catch: (error) =>
      workerRuntimeErrorFromUnknown(
        "model_asset_store.cleanup_opfs_package",
        error,
        `failed to clean OPFS model asset payload ${area}/${ref}`,
      ),
  }).pipe(Effect.ignore);
}

function makeModelAssetCatalog(): Effect.Effect<
  ModelAssetCatalogApi,
  WorkerRuntimeError,
  IndexedDbDatabase.IndexedDbDatabase
> {
  return Effect.gen(function*() {
    const database = yield* IndexedDbDatabase.IndexedDbDatabase;
    const clock = yield* Clock.Clock;

    const decodePackageRecord = Schema.decodeUnknownEffect(
      ModelAssetPackageRecordSchema,
    );
    const decodeBlobRecord = Schema.decodeUnknownEffect(
      ModelAssetBlobRecordSchema,
    );

    const getRecord = Effect.fn("ModelAssetCatalog.getRecord")(function*(packageId: string) {
      const transaction = database.database.current.transaction(
        ["packages"],
        "readonly",
      );
      const row = yield* runTransactionCallback<unknown | null>({
        transaction,
        operation: "model_asset_catalog.get_record",
        failureMessage: `failed to read model asset package record ${packageId}`,
        register: ({ registerRequest, succeed, fail }) => {
          const request = registerRequest(
            transaction.objectStore("packages").get(packageId),
          );
          request.onerror = (event) => {
            fail(request.error ?? event);
          };
          request.onsuccess = () => {
            succeed(request.result ?? null);
          };
        },
      });

      if (row === null) {
        return null;
      }

      return yield* decodePackageRecord(row).pipe(
        Effect.mapError((error) =>
          workerRuntimeErrorFromUnknown(
            "model_asset_catalog.decode_record",
            error,
            `failed to decode model asset package record ${packageId}`,
          ),
        ),
      );
    });

    const putRecord = Effect.fn("ModelAssetCatalog.putRecord")(function*(
      record: ModelAssetPackageRecord,
    ) {
      const transaction = database.database.current.transaction(
        ["packages"],
        "readwrite",
      );
      yield* runTransactionCallback<void>({
        transaction,
        operation: "model_asset_catalog.put_record",
        failureMessage: `failed to upsert model asset package record ${record.packageId}`,
        register: ({ registerRequest, succeed, fail }) => {
          const request = registerRequest(
            transaction.objectStore("packages").put(record),
          );
          request.onerror = (event) => {
            fail(request.error ?? event);
          };
          transaction.oncomplete = () => {
            succeed(void 0);
          };
        },
      });
    });

    const touchRecord = Effect.fn("ModelAssetCatalog.touchRecord")(function*(
      packageId: string,
      lastAccessedAtMs: number,
    ) {
      const record = yield* getRecord(packageId);
      if (record === null) {
        return;
      }
      yield* putRecord({
        ...record,
        lastAccessedAtMs,
      });
    });

    const removePackage = Effect.fn("ModelAssetCatalog.removePackage")(function*(
      packageId: string,
    ) {
      const transaction = database.database.current.transaction(
        ["packages", "asset_blobs"],
        "readwrite",
      );
      yield* runTransactionCallback<void>({
        transaction,
        operation: "model_asset_catalog.remove_package",
        failureMessage: `failed to remove model asset package ${packageId}`,
        register: ({ registerRequest, succeed }) => {
          const packagesStore = transaction.objectStore("packages");
          const blobsStore = transaction.objectStore("asset_blobs");

          for (const assetKind of MODEL_ASSET_KINDS) {
            registerRequest(blobsStore.delete([packageId, assetKind]));
          }
          registerRequest(packagesStore.delete(packageId));

          transaction.oncomplete = () => {
            succeed(void 0);
          };
        },
      });
    });

    const loadIndexedDbPackage = Effect.fn(
      "ModelAssetCatalog.loadIndexedDbPackage",
    )(function*(key: ModelAssetPackageKey) {
      const transaction = database.database.current.transaction(
        ["asset_blobs"],
        "readonly",
      );
      const rawRows = yield* runTransactionCallback<Array<unknown>>({
        transaction,
        operation: "model_asset_catalog.load_indexeddb_package",
        failureMessage: `failed to read stored asset blobs for ${key.packageId}`,
        register: ({ registerRequest, succeed, fail }) => {
          const index = transaction.objectStore("asset_blobs").index("byPackage");
          const request = registerRequest(index.getAll(key.packageId));
          request.onerror = (event) => {
            fail(request.error ?? event);
          };
          request.onsuccess = () => {
            succeed(Array.from(request.result ?? []));
          };
        },
      });

      const rows = yield* Effect.forEach(rawRows, (row) =>
        decodeBlobRecord(row).pipe(
          Effect.mapError((error) =>
            workerRuntimeErrorFromUnknown(
              "model_asset_catalog.decode_blob",
              error,
              `failed to decode stored asset blob for ${key.packageId}`,
            ),
          ),
        ),
      );

      const bytesByKind = new Map<ModelAssetKind, Uint8Array>();
      for (const row of rows) {
        bytesByKind.set(row.assetKind, row.bytes);
      }

      return packageFromBytes(key, bytesByKind);
    });

    const commitIndexedDbPackage = Effect.fn(
      "ModelAssetCatalog.commitIndexedDbPackage",
    )(function*(record: ModelAssetPackageRecord, pkg: ModelAssetPackage) {
      const writeTimestamp = clock.currentTimeMillisUnsafe();
      const readyRecord = {
        ...record,
        state: "ready" as const,
        backend: "indexeddb" as const,
        payloadRef: pkg.key.packageId,
        lastAccessedAtMs: writeTimestamp,
      };

      const transaction = database.database.current.transaction(
        ["packages", "asset_blobs"],
        "readwrite",
      );
      yield* runTransactionCallback<void>({
        transaction,
        operation: "model_asset_catalog.commit_indexeddb_package",
        failureMessage: `failed to commit indexeddb model asset package ${pkg.key.packageId}`,
        register: ({ registerRequest, succeed }) => {
          const packagesStore = transaction.objectStore("packages");
          const blobsStore = transaction.objectStore("asset_blobs");

          for (const assetKind of MODEL_ASSET_KINDS) {
            registerRequest(
              blobsStore.put({
                packageId: pkg.key.packageId,
                assetKind,
                bytes: modelAssetBytes(pkg, assetKind),
              }),
            );
          }
          registerRequest(packagesStore.put(readyRecord));

          transaction.oncomplete = () => {
            succeed(void 0);
          };
        },
      });
    });

    return ModelAssetCatalog.of({
      getRecord,
      putRecord,
      touchRecord,
      removePackage,
      loadIndexedDbPackage,
      commitIndexedDbPackage,
    });
  });
}

function makeModelAssetOpfsStore(): Effect.Effect<
  ModelAssetOpfsStoreApi,
  WorkerRuntimeError
> {
  return Effect.gen(function*() {
    if (!supportsOpfs()) {
      return yield* workerRuntimeError({
        operation: "model_asset_opfs_store.init",
        message: "OPFS is unavailable in this worker",
      });
    }

    const opfsRoot = yield* Effect.tryPromise({
      try: async () => {
        const root = await navigator.storage.getDirectory();
        return root.getDirectoryHandle(MODEL_ASSET_OPFS_ROOT_DIR, {
          create: true,
        });
      },
      catch: (error) =>
        workerRuntimeErrorFromUnknown(
          "model_asset_opfs_store.root",
          error,
          "failed to open model asset OPFS root",
        ),
    });

    const getAreaDirectory = (
      area: ModelAssetArea,
      create: boolean,
    ): Effect.Effect<FileSystemDirectoryHandle | null, WorkerRuntimeError> =>
      Effect.tryPromise({
        try: async () => {
          try {
            return await opfsRoot.getDirectoryHandle(area, { create });
          } catch (error) {
            if (
              error instanceof DOMException &&
              error.name === "NotFoundError" &&
              !create
            ) {
              return null;
            }
            throw error;
          }
        },
        catch: (error) =>
          workerRuntimeErrorFromUnknown(
            "model_asset_opfs_store.area_directory",
            error,
            `failed to open OPFS model asset area ${area}`,
          ),
      });

    const getPackageDirectory = (
      area: ModelAssetArea,
      ref: string,
      create: boolean,
    ): Effect.Effect<FileSystemDirectoryHandle | null, WorkerRuntimeError> =>
      Effect.gen(function*() {
        const areaDirectory = yield* getAreaDirectory(area, create);
        if (areaDirectory === null) {
          return null;
        }

        return yield* Effect.tryPromise({
          try: async () => {
            try {
              return await areaDirectory.getDirectoryHandle(ref, { create });
            } catch (error) {
              if (
                error instanceof DOMException &&
                error.name === "NotFoundError" &&
                !create
              ) {
                return null;
              }
              throw error;
            }
          },
          catch: (error) =>
            workerRuntimeErrorFromUnknown(
              "model_asset_opfs_store.package_directory",
              error,
              `failed to open OPFS package directory ${area}/${ref}`,
            ),
        });
      });

    const readFileBytes = (
      directory: FileSystemDirectoryHandle,
      fileName: string,
    ): Effect.Effect<Uint8Array | null, WorkerRuntimeError> =>
      Effect.tryPromise({
        try: async () => {
          try {
            const handle = await directory.getFileHandle(fileName, {
              create: false,
            });
            const file = await handle.getFile();
            return new Uint8Array(await file.arrayBuffer());
          } catch (error) {
            if (
              error instanceof DOMException &&
              error.name === "NotFoundError"
            ) {
              return null;
            }
            throw error;
          }
        },
        catch: (error) =>
          workerRuntimeErrorFromUnknown(
            "model_asset_opfs_store.read_file",
            error,
            `failed to read OPFS file ${fileName}`,
          ),
      });

    const writeFileBytes = (
      directory: FileSystemDirectoryHandle,
      fileName: string,
      bytes: Uint8Array,
    ): Effect.Effect<void, WorkerRuntimeError> =>
      Effect.tryPromise({
        try: async () => {
          const handle = await directory.getFileHandle(fileName, { create: true });
          const writable = await handle.createWritable();
          await writable.write(new Uint8Array(bytes));
          await writable.close();
        },
        catch: (error) =>
          workerRuntimeErrorFromUnknown(
            "model_asset_opfs_store.write_file",
            error,
            `failed to write OPFS file ${fileName}`,
          ),
      });

    const removePackage = Effect.fn("ModelAssetOpfsStore.removePackage")(function*(
      area: ModelAssetArea,
      ref: string,
    ) {
      const areaDirectory = yield* getAreaDirectory(area, false);
      if (areaDirectory === null) {
        return;
      }

      yield* Effect.tryPromise({
        try: async () => {
          try {
            await areaDirectory.removeEntry(ref, { recursive: true });
          } catch (error) {
            if (
              error instanceof DOMException &&
              error.name === "NotFoundError"
            ) {
              return;
            }
            throw error;
          }
        },
        catch: (error) =>
          workerRuntimeErrorFromUnknown(
            "model_asset_opfs_store.remove_package",
            error,
            `failed to remove OPFS package ${area}/${ref}`,
          ),
      });
    });

    const loadPackage = Effect.fn("ModelAssetOpfsStore.loadPackage")(function*(
      area: ModelAssetArea,
      ref: string,
      key: ModelAssetPackageKey,
    ) {
      const directory = yield* getPackageDirectory(area, ref, false);
      if (directory === null) {
        return null;
      }

      const modelBytes = yield* readFileBytes(directory, "model.onnx");
      const tokenizerBytes = yield* readFileBytes(directory, "tokenizer.json");
      const onnxConfigBytes = yield* readFileBytes(directory, "onnx_config.json");

      if (
        modelBytes === null ||
        tokenizerBytes === null ||
        onnxConfigBytes === null
      ) {
        return null;
      }

      return {
        key,
        modelBytes,
        tokenizerBytes,
        onnxConfigBytes,
      } satisfies ModelAssetPackage;
    });

    const writePackage = Effect.fn("ModelAssetOpfsStore.writePackage")(function*(
      area: ModelAssetArea,
      ref: string,
      pkg: ModelAssetPackage,
    ) {
      const directory = yield* getPackageDirectory(area, ref, true);
      if (directory === null) {
        return yield* workerRuntimeError({
          operation: "model_asset_opfs_store.write_package",
          message: `failed to create OPFS package directory ${area}/${ref}`,
        });
      }

      yield* writeFileBytes(directory, "model.onnx", pkg.modelBytes);
      yield* writeFileBytes(directory, "tokenizer.json", pkg.tokenizerBytes);
      yield* writeFileBytes(directory, "onnx_config.json", pkg.onnxConfigBytes);
    });

    return ModelAssetOpfsStore.of({
      loadPackage,
      writePackage,
      removePackage,
    });
  });
}

function makeOpfsModelAssetStore(): Effect.Effect<
  ModelAssetStoreApi,
  WorkerRuntimeError,
  ModelAssetCatalog | ModelAssetOpfsStore
> {
  return Effect.gen(function*() {
    const catalog = yield* ModelAssetCatalog;
    const opfsStore = yield* ModelAssetOpfsStore;
    const clock = yield* Clock.Clock;

    const loadPackage = Effect.fn("ModelAssetStore.loadPackage.opfs")(function*(
      key: ModelAssetPackageKey,
    ) {
      const record = yield* catalog.getRecord(key.packageId);
      if (record === null) {
        return null;
      }

      if (record.backend !== "opfs") {
        yield* catalog.removePackage(key.packageId).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        yield* opfsStore.removePackage("packages", key.packageId).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        yield* opfsStore.removePackage("staging", record.payloadRef).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        return null;
      }

      if (record.state === "staging") {
        yield* catalog.removePackage(key.packageId).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        yield* opfsStore.removePackage("staging", record.payloadRef).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        yield* opfsStore.removePackage("packages", key.packageId).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        return null;
      }

      const pkg = yield* opfsStore.loadPackage("packages", record.payloadRef, key);
      if (pkg === null) {
        yield* catalog.removePackage(key.packageId).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        yield* opfsStore.removePackage("packages", record.payloadRef).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        yield* opfsStore.removePackage("staging", record.payloadRef).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        return null;
      }

      yield* catalog.touchRecord(
        key.packageId,
        clock.currentTimeMillisUnsafe(),
      ).pipe(Effect.orElseSucceed(() => void 0));

      return pkg;
    });

    const storePackage = Effect.fn("ModelAssetStore.storePackage.opfs")(function*(
      pkg: ModelAssetPackage,
    ) {
      const now = clock.currentTimeMillisUnsafe();
      const readyRecord = modelAssetRecord(pkg.key, {
        state: "ready",
        backend: "opfs",
        payloadRef: pkg.key.packageId,
        createdAtMs: now,
        lastAccessedAtMs: now,
      });

      yield* opfsStore.writePackage("packages", pkg.key.packageId, pkg).pipe(
        Effect.flatMap(() => catalog.putRecord(readyRecord)),
        Effect.catchTag("WorkerRuntimeError", (error) =>
          Effect.gen(function*() {
            yield* opfsStore.removePackage("packages", pkg.key.packageId).pipe(
              Effect.orElseSucceed(() => void 0),
            );
            yield* catalog.removePackage(pkg.key.packageId).pipe(
              Effect.orElseSucceed(() => void 0),
            );
            return yield* error;
          }),
        ),
      );
    });

    const removePackage = Effect.fn("ModelAssetStore.removePackage.opfs")(function*(
      packageId: string,
    ) {
      const record = yield* catalog.getRecord(packageId);
      yield* catalog.removePackage(packageId).pipe(
        Effect.orElseSucceed(() => void 0),
      );
      yield* opfsStore.removePackage("packages", packageId).pipe(
        Effect.orElseSucceed(() => void 0),
      );
      if (record !== null && record.state === "staging") {
        yield* opfsStore.removePackage("staging", record.payloadRef).pipe(
          Effect.orElseSucceed(() => void 0),
        );
      }
    });

    return ModelAssetStore.of({
      loadPackage,
      storePackage,
      removePackage,
      persistentStorage: () => Effect.succeed(true),
      kind: () => Effect.succeed("opfs"),
    });
  });
}

function makeIndexedDbModelAssetStore(): Effect.Effect<
  ModelAssetStoreApi,
  WorkerRuntimeError,
  ModelAssetCatalog
> {
  return Effect.gen(function*() {
    const catalog = yield* ModelAssetCatalog;
    const clock = yield* Clock.Clock;

    const loadPackage = Effect.fn("ModelAssetStore.loadPackage.indexeddb")(function*(
      key: ModelAssetPackageKey,
    ) {
      const record = yield* catalog.getRecord(key.packageId);
      if (record === null) {
        return null;
      }

      if (record.backend !== "indexeddb" || record.state !== "ready") {
        yield* catalog.removePackage(key.packageId).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        if (record.backend === "opfs") {
          yield* removeOpfsPackageIfAvailable("packages", key.packageId);
          if (record.payloadRef !== key.packageId) {
            yield* removeOpfsPackageIfAvailable("staging", record.payloadRef);
          }
        }
        return null;
      }

      const pkg = yield* catalog.loadIndexedDbPackage(key);
      if (pkg === null) {
        yield* catalog.removePackage(key.packageId).pipe(
          Effect.orElseSucceed(() => void 0),
        );
        return null;
      }

      yield* catalog.touchRecord(
        key.packageId,
        clock.currentTimeMillisUnsafe(),
      ).pipe(Effect.orElseSucceed(() => void 0));

      return pkg;
    });

    const storePackage = Effect.fn(
      "ModelAssetStore.storePackage.indexeddb",
    )(function*(pkg: ModelAssetPackage) {
      const now = clock.currentTimeMillisUnsafe();
      const record = modelAssetRecord(pkg.key, {
        state: "ready",
        backend: "indexeddb",
        payloadRef: pkg.key.packageId,
        createdAtMs: now,
        lastAccessedAtMs: now,
      });

      yield* catalog.commitIndexedDbPackage(record, pkg);
    });

    const removePackage = Effect.fn(
      "ModelAssetStore.removePackage.indexeddb",
    )(function*(packageId: string) {
      yield* catalog.removePackage(packageId).pipe(
        Effect.orElseSucceed(() => void 0),
      );
    });

    return ModelAssetStore.of({
      loadPackage,
      storePackage,
      removePackage,
      persistentStorage: () => Effect.succeed(true),
      kind: () => Effect.succeed("indexeddb"),
    });
  });
}

function makeTransientModelAssetStore(): Effect.Effect<
  ModelAssetStoreApi,
  never
> {
  return Effect.gen(function*() {
    const storeRef = yield* Ref.make(new Map<string, ModelAssetPackage>());

    const loadPackage = Effect.fn("ModelAssetStore.loadPackage.transient")(function*(
      key: ModelAssetPackageKey,
    ) {
      return yield* Ref.get(storeRef).pipe(
        Effect.map((store) => store.get(key.packageId) ?? null),
      );
    });

    const storePackage = Effect.fn("ModelAssetStore.storePackage.transient")(function*(
      pkg: ModelAssetPackage,
    ) {
      yield* Ref.update(storeRef, (store) => {
        const next = new Map(store);
        next.set(pkg.key.packageId, pkg);
        return next;
      });
    });

    const removePackage = Effect.fn("ModelAssetStore.removePackage.transient")(function*(
      packageId: string,
    ) {
      yield* Ref.update(storeRef, (store) => {
        if (!store.has(packageId)) {
          return store;
        }
        const next = new Map(store);
        next.delete(packageId);
        return next;
      });
    });

    return ModelAssetStore.of({
      loadPackage,
      storePackage,
      removePackage,
      persistentStorage: () => Effect.succeed(false),
      kind: () => Effect.succeed("transient"),
    });
  });
}
