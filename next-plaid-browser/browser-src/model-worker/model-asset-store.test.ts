import { expect, it } from "@effect/vitest";
import { IDBFactory, IDBKeyRange } from "fake-indexeddb";
import { Context, Effect, Layer, Scope } from "effect";
import * as Exit from "effect/Exit";

import {
  type WorkerRuntimeError,
  workerRuntimeErrorFromUnknown,
} from "../effect/worker-runtime-errors.js";
import { ModelAssetStore } from "./model-asset-store.js";
import {
  MODEL_ASSET_DB_NAME,
  MODEL_ASSET_OPFS_ROOT_DIR,
  type ModelAssetKind,
  type ModelAssetPackage,
} from "./model-asset-types.js";

class FakeOpfsFile {
  bytes = new Uint8Array();
}

class FakeOpfsFileHandle {
  constructor(private readonly file: FakeOpfsFile) {}

  async getFile(): Promise<File> {
    const snapshot = this.file.bytes.slice();
    return {
      arrayBuffer: async () =>
        snapshot.buffer.slice(
          snapshot.byteOffset,
          snapshot.byteOffset + snapshot.byteLength,
        ),
    } as File;
  }

  async createWritable(): Promise<FileSystemWritableFileStream> {
    const file = this.file;
    return {
      write: async (data: BufferSource | Blob | string) => {
        if (typeof data === "string" || data instanceof Blob) {
          throw new Error("unexpected OPFS write payload in test");
        }
        file.bytes =
          data instanceof Uint8Array
            ? data.slice()
            : new Uint8Array(data as ArrayBuffer);
      },
      close: async () => {},
    } as FileSystemWritableFileStream;
  }
}

class FakeOpfsDirectoryHandle {
  readonly #directories = new Map<string, FakeOpfsDirectoryHandle>();
  readonly #files = new Map<string, FakeOpfsFile>();

  constructor(readonly name: string) {}

  async getDirectoryHandle(
    name: string,
    options?: FileSystemGetDirectoryOptions,
  ): Promise<FileSystemDirectoryHandle> {
    const existing = this.#directories.get(name);
    if (existing !== undefined) {
      return existing as unknown as FileSystemDirectoryHandle;
    }
    if (options?.create === true) {
      const created = new FakeOpfsDirectoryHandle(name);
      this.#directories.set(name, created);
      return created as unknown as FileSystemDirectoryHandle;
    }
    throw new DOMException(
      `directory ${name} was not found`,
      "NotFoundError",
    );
  }

  async getFileHandle(
    name: string,
    options?: FileSystemGetFileOptions,
  ): Promise<FileSystemFileHandle> {
    const existing = this.#files.get(name);
    if (existing !== undefined) {
      return new FakeOpfsFileHandle(existing) as unknown as FileSystemFileHandle;
    }
    if (options?.create === true) {
      const created = new FakeOpfsFile();
      this.#files.set(name, created);
      return new FakeOpfsFileHandle(created) as unknown as FileSystemFileHandle;
    }
    throw new DOMException(`file ${name} was not found`, "NotFoundError");
  }

  async removeEntry(name: string): Promise<void> {
    if (this.#directories.delete(name)) {
      return;
    }
    if (this.#files.delete(name)) {
      return;
    }
    throw new DOMException(`entry ${name} was not found`, "NotFoundError");
  }

  directory(name: string): FakeOpfsDirectoryHandle | undefined {
    return this.#directories.get(name);
  }

  file(name: string): FakeOpfsFile | undefined {
    return this.#files.get(name);
  }
}

class FakeStorageManager {
  readonly #root = new FakeOpfsDirectoryHandle("root");

  constructor(private readonly persistedValue: boolean) {}

  async getDirectory(): Promise<FileSystemDirectoryHandle> {
    return this.#root as unknown as FileSystemDirectoryHandle;
  }

  async persisted(): Promise<boolean> {
    return this.persistedValue;
  }

  hasPackageDirectory(packageId: string): boolean {
    return this.#root
      .directory(MODEL_ASSET_OPFS_ROOT_DIR)
      ?.directory("packages")
      ?.directory(packageId) !== undefined;
  }

  removePackageFile(packageId: string, fileName: string): void {
    const packagesDirectory = this.#root
      .directory(MODEL_ASSET_OPFS_ROOT_DIR)
      ?.directory("packages")
      ?.directory(packageId);
    if (packagesDirectory === undefined) {
      throw new Error(`missing fake OPFS package ${packageId}`);
    }
    if (packagesDirectory.file(fileName) === undefined) {
      throw new Error(
        `missing fake OPFS package file ${packageId}/${fileName}`,
      );
    }
    packagesDirectory.removeEntry(fileName);
  }
}

class RejectingStorageManager {
  async getDirectory(): Promise<FileSystemDirectoryHandle> {
    throw new DOMException("OPFS blocked by browser policy", "NotAllowedError");
  }

  async estimate(): Promise<StorageEstimate> {
    return {};
  }

  async persist(): Promise<boolean> {
    return false;
  }

  async persisted(): Promise<boolean> {
    return false;
  }
}

function failingIndexedDbFactory(): IDBFactory {
  return {
    open() {
      throw new DOMException("IndexedDB blocked by browser policy", "InvalidStateError");
    },
  } as unknown as IDBFactory;
}

function modelAssetPackage(packageId = "pkg-1"): ModelAssetPackage {
  return {
    key: {
      packageId,
      encoderId: "proof-encoder",
      encoderBuild: "build-1",
      embeddingDim: 4,
      normalized: true,
      modelUrl: "https://example.test/model.onnx",
      tokenizerUrl: "https://example.test/tokenizer.json",
      onnxConfigUrl: "https://example.test/onnx_config.json",
    },
    modelBytes: new Uint8Array([1, 2, 3]),
    tokenizerBytes: new Uint8Array([4, 5, 6]),
    onnxConfigBytes: new Uint8Array([7, 8, 9]),
  };
}

function patchBrowserGlobals(options: {
  readonly indexedDb: boolean;
  readonly opfs: boolean;
  readonly persisted?: boolean;
  readonly indexedDbFactory?: IDBFactory;
  readonly storageOverride?: StorageManager | null;
}): {
  readonly restore: () => void;
  readonly storage: FakeStorageManager | null;
} {
  const originalIndexedDb = (globalThis as Record<string, unknown>).indexedDB;
  const originalIDBKeyRange = (globalThis as Record<string, unknown>).IDBKeyRange;
  const navigatorObject = globalThis.navigator as Navigator & {
    storage?: StorageManager;
  };
  const originalStorage = navigatorObject.storage;
  const storage =
    options.storageOverride !== undefined
      ? options.storageOverride
      : options.opfs === true
      ? new FakeStorageManager(options.persisted ?? true)
      : null;
  const indexedDb =
    options.indexedDbFactory ??
    (options.indexedDb ? new IDBFactory() : undefined);

  Object.defineProperty(globalThis, "indexedDB", {
    configurable: true,
    value: indexedDb,
  });
  Object.defineProperty(globalThis, "IDBKeyRange", {
    configurable: true,
    value: options.indexedDb ? IDBKeyRange : undefined,
  });
  Object.defineProperty(navigatorObject, "storage", {
    configurable: true,
    value: storage ?? undefined,
  });

  return {
    restore: () => {
      Object.defineProperty(globalThis, "indexedDB", {
        configurable: true,
        value: originalIndexedDb,
      });
      Object.defineProperty(globalThis, "IDBKeyRange", {
        configurable: true,
        value: originalIDBKeyRange,
      });
      Object.defineProperty(navigatorObject, "storage", {
        configurable: true,
        value: originalStorage,
      });
    },
    storage: storage instanceof FakeStorageManager ? storage : null,
  };
}

function buildStore(
  scope: Scope.Scope,
  layer: Layer.Layer<ModelAssetStore, unknown>,
): Effect.Effect<Context.Context<ModelAssetStore>> {
  return Layer.buildWithScope(layer, scope) as Effect.Effect<
    Context.Context<ModelAssetStore>,
    never
  >;
}

function openRawDatabase(): Effect.Effect<IDBDatabase, WorkerRuntimeError> {
  return Effect.callback<IDBDatabase, WorkerRuntimeError>((resume) => {
    const request = globalThis.indexedDB.open(MODEL_ASSET_DB_NAME);

    request.onerror = (event) => {
      resume(
        Effect.fail(
          workerRuntimeErrorFromUnknown(
            "model_asset_store_test.open_raw_database",
            request.error ?? event,
            "failed to open raw IndexedDB database",
          ),
        ),
      );
    };
    request.onsuccess = () => {
      resume(Effect.succeed(request.result));
    };
  });
}

function withRawDatabase<A>(
  use: (database: IDBDatabase) => Effect.Effect<A, WorkerRuntimeError>,
): Effect.Effect<A, WorkerRuntimeError> {
  return Effect.acquireUseRelease(
    openRawDatabase(),
    use,
    (database) => Effect.sync(() => database.close()),
  );
}

function deleteRawAssetBlob(
  packageId: string,
  assetKind: ModelAssetKind,
): Effect.Effect<void, WorkerRuntimeError> {
  return withRawDatabase((database) =>
    Effect.callback<void, WorkerRuntimeError>((resume) => {
      const transaction = database.transaction(["asset_blobs"], "readwrite");
      const request = transaction
        .objectStore("asset_blobs")
        .delete([packageId, assetKind]);

      request.onerror = (event) => {
        resume(
          Effect.fail(
            workerRuntimeErrorFromUnknown(
              "model_asset_store_test.delete_raw_asset_blob",
              request.error ?? event,
              "failed to delete raw asset blob",
            ),
          ),
        );
      };
      transaction.onerror = (event) => {
        resume(
          Effect.fail(
            workerRuntimeErrorFromUnknown(
              "model_asset_store_test.delete_raw_asset_blob",
              transaction.error ?? event,
              "failed to commit raw asset blob delete",
            ),
          ),
        );
      };
      transaction.oncomplete = () => {
        resume(Effect.void);
      };
    })
  );
}

function readRawPackageRecord(
  packageId: string,
): Effect.Effect<unknown | null, WorkerRuntimeError> {
  return withRawDatabase((database) =>
    Effect.callback<unknown | null, WorkerRuntimeError>((resume) => {
      const transaction = database.transaction(["packages"], "readonly");
      const request = transaction.objectStore("packages").get(packageId);

      request.onerror = (event) => {
        resume(
          Effect.fail(
            workerRuntimeErrorFromUnknown(
              "model_asset_store_test.read_raw_package_record",
              request.error ?? event,
              "failed to read raw package record",
            ),
          ),
        );
      };
      request.onsuccess = () => {
        resume(Effect.succeed(request.result ?? null));
      };
    })
  );
}

it.effect("reopens an OPFS-backed package across service lifetimes", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: true,
      opfs: true,
      persisted: false,
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const packageToStore = modelAssetPackage("pkg-opfs");

    const firstScope = yield* Scope.make();
    const firstContext = yield* buildStore(firstScope, ModelAssetStore.layerOpfs);
    const firstStore = Context.get(firstContext, ModelAssetStore);
    yield* firstStore.storePackage(packageToStore);
    expect(yield* firstStore.kind()).toBe("opfs");
    expect(yield* firstStore.persistentStorage()).toBe(true);
    expect(yield* firstStore.loadPackage(packageToStore.key)).toEqual(
      packageToStore,
    );
    yield* Scope.close(firstScope, Exit.void);

    const secondScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(secondScope, Exit.void));
    const secondContext = yield* buildStore(
      secondScope,
      ModelAssetStore.layerOpfs,
    );
    const secondStore = Context.get(secondContext, ModelAssetStore);

    expect(yield* secondStore.loadPackage(packageToStore.key)).toEqual(
      packageToStore,
    );
  }),
);

it.effect("reopens an IndexedDB-backed package across service lifetimes", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: true,
      opfs: false,
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const packageToStore = modelAssetPackage("pkg-indexeddb");

    const firstScope = yield* Scope.make();
    const firstContext = yield* buildStore(
      firstScope,
      ModelAssetStore.layerIndexedDb,
    );
    const firstStore = Context.get(firstContext, ModelAssetStore);
    yield* firstStore.storePackage(packageToStore);
    expect(yield* firstStore.kind()).toBe("indexeddb");
    expect(yield* firstStore.persistentStorage()).toBe(true);
    expect(yield* firstStore.loadPackage(packageToStore.key)).toEqual(
      packageToStore,
    );
    yield* Scope.close(firstScope, Exit.void);

    const secondScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(secondScope, Exit.void));
    const secondContext = yield* buildStore(
      secondScope,
      ModelAssetStore.layerIndexedDb,
    );
    const secondStore = Context.get(secondContext, ModelAssetStore);

    expect(yield* secondStore.loadPackage(packageToStore.key)).toEqual(
      packageToStore,
    );
  }),
);

it.effect("falls back from OPFS runtime failure to IndexedDB in auto mode", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: true,
      opfs: false,
      storageOverride: new RejectingStorageManager(),
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const packageToStore = modelAssetPackage("pkg-auto-indexeddb");
    const scope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(scope, Exit.void));
    const context = yield* buildStore(scope, ModelAssetStore.layerAuto);
    const store = Context.get(context, ModelAssetStore);

    expect(yield* store.kind()).toBe("indexeddb");
    expect(yield* store.persistentStorage()).toBe(true);
    yield* store.storePackage(packageToStore);
    expect(yield* store.loadPackage(packageToStore.key)).toEqual(packageToStore);
  }),
);

it.effect("falls back to transient when both durable backends fail at runtime", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: true,
      opfs: false,
      indexedDbFactory: failingIndexedDbFactory(),
      storageOverride: new RejectingStorageManager(),
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const packageToStore = modelAssetPackage("pkg-auto-transient");

    const firstScope = yield* Scope.make();
    const firstContext = yield* buildStore(firstScope, ModelAssetStore.layerAuto);
    const firstStore = Context.get(firstContext, ModelAssetStore);
    expect(yield* firstStore.kind()).toBe("transient");
    expect(yield* firstStore.persistentStorage()).toBe(false);
    yield* firstStore.storePackage(packageToStore);
    expect(yield* firstStore.loadPackage(packageToStore.key)).toEqual(
      packageToStore,
    );
    yield* Scope.close(firstScope, Exit.void);

    const secondScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(secondScope, Exit.void));
    const secondContext = yield* buildStore(
      secondScope,
      ModelAssetStore.layerAuto,
    );
    const secondStore = Context.get(secondContext, ModelAssetStore);
    expect(yield* secondStore.kind()).toBe("transient");
    expect(yield* secondStore.loadPackage(packageToStore.key)).toBeNull();
  }),
);

it.effect("prefers OPFS, then IndexedDB, then transient in auto mode", () =>
  Effect.gen(function* () {
    const bothGlobals = patchBrowserGlobals({
      indexedDb: true,
      opfs: true,
    });
    yield* Effect.addFinalizer(() => Effect.sync(bothGlobals.restore));
    const bothScope = yield* Scope.make();
    const bothContext = yield* buildStore(bothScope, ModelAssetStore.layerAuto);
    expect(yield* Context.get(bothContext, ModelAssetStore).kind()).toBe(
      "opfs",
    );
    yield* Scope.close(bothScope, Exit.void);
    bothGlobals.restore();

    const indexedDbGlobals = patchBrowserGlobals({
      indexedDb: true,
      opfs: false,
    });
    const indexedDbScope = yield* Scope.make();
    const indexedDbContext = yield* buildStore(
      indexedDbScope,
      ModelAssetStore.layerAuto,
    );
    expect(
      yield* Context.get(indexedDbContext, ModelAssetStore).kind(),
    ).toBe("indexeddb");
    yield* Scope.close(indexedDbScope, Exit.void);
    indexedDbGlobals.restore();

    const transientGlobals = patchBrowserGlobals({
      indexedDb: false,
      opfs: false,
    });
    const transientScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(transientScope, Exit.void));
    yield* Effect.addFinalizer(() => Effect.sync(transientGlobals.restore));
    const transientContext = yield* buildStore(
      transientScope,
      ModelAssetStore.layerAuto,
    );
    expect(
      yield* Context.get(transientContext, ModelAssetStore).kind(),
    ).toBe("transient");
  }),
);

it.effect("falls back to transient storage and loses the package after reload", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: false,
      opfs: false,
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const packageToStore = modelAssetPackage("pkg-transient");

    const firstScope = yield* Scope.make();
    const firstContext = yield* buildStore(
      firstScope,
      ModelAssetStore.layerAuto,
    );
    const firstStore = Context.get(firstContext, ModelAssetStore);
    yield* firstStore.storePackage(packageToStore);
    expect(yield* firstStore.kind()).toBe("transient");
    expect(yield* firstStore.loadPackage(packageToStore.key)).toEqual(
      packageToStore,
    );
    yield* Scope.close(firstScope, Exit.void);

    const secondScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(secondScope, Exit.void));
    const secondContext = yield* buildStore(
      secondScope,
      ModelAssetStore.layerAuto,
    );
    const secondStore = Context.get(secondContext, ModelAssetStore);

    expect(yield* secondStore.loadPackage(packageToStore.key)).toBeNull();
  }),
);

it.effect("cleans up a corrupt ready IndexedDB package and drops its catalog record", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: true,
      opfs: false,
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const packageToStore = modelAssetPackage("pkg-indexeddb-corrupt");

    const firstScope = yield* Scope.make();
    const firstContext = yield* buildStore(
      firstScope,
      ModelAssetStore.layerIndexedDb,
    );
    const firstStore = Context.get(firstContext, ModelAssetStore);
    yield* firstStore.storePackage(packageToStore);
    yield* Scope.close(firstScope, Exit.void);

    yield* deleteRawAssetBlob(packageToStore.key.packageId, "tokenizer");

    const secondScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(secondScope, Exit.void));
    const secondContext = yield* buildStore(
      secondScope,
      ModelAssetStore.layerIndexedDb,
    );
    const secondStore = Context.get(secondContext, ModelAssetStore);

    expect(yield* secondStore.loadPackage(packageToStore.key)).toBeNull();
    expect(
      yield* readRawPackageRecord(packageToStore.key.packageId),
    ).toBeNull();
  }),
);

it.effect("cleans up stale OPFS payloads when the IndexedDB path encounters an OPFS record", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: true,
      opfs: true,
      persisted: true,
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const storage = globals.storage;
    expect(storage).not.toBeNull();
    if (storage === null) {
      return;
    }

    const packageToStore = modelAssetPackage("pkg-cross-backend");

    const firstScope = yield* Scope.make();
    const firstContext = yield* buildStore(firstScope, ModelAssetStore.layerOpfs);
    const firstStore = Context.get(firstContext, ModelAssetStore);
    yield* firstStore.storePackage(packageToStore);
    yield* Scope.close(firstScope, Exit.void);

    expect(storage.hasPackageDirectory(packageToStore.key.packageId)).toBe(true);

    const secondScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(secondScope, Exit.void));
    const secondContext = yield* buildStore(
      secondScope,
      ModelAssetStore.layerIndexedDb,
    );
    const secondStore = Context.get(secondContext, ModelAssetStore);

    expect(yield* secondStore.loadPackage(packageToStore.key)).toBeNull();
    expect(storage.hasPackageDirectory(packageToStore.key.packageId)).toBe(
      false,
    );
    expect(
      yield* readRawPackageRecord(packageToStore.key.packageId),
    ).toBeNull();
  }),
);

it.effect("cleans up a corrupt ready OPFS package and treats it as a miss", () =>
  Effect.gen(function* () {
    const globals = patchBrowserGlobals({
      indexedDb: true,
      opfs: true,
      persisted: true,
    });
    yield* Effect.addFinalizer(() => Effect.sync(globals.restore));

    const storage = globals.storage;
    expect(storage).not.toBeNull();
    if (storage === null) {
      return;
    }

    const packageToStore = modelAssetPackage("pkg-corrupt");

    const firstScope = yield* Scope.make();
    const firstContext = yield* buildStore(firstScope, ModelAssetStore.layerOpfs);
    const firstStore = Context.get(firstContext, ModelAssetStore);
    yield* firstStore.storePackage(packageToStore);
    yield* Scope.close(firstScope, Exit.void);

    storage.removePackageFile(packageToStore.key.packageId, "tokenizer.json");
    expect(storage.hasPackageDirectory(packageToStore.key.packageId)).toBe(true);

    const secondScope = yield* Scope.make();
    yield* Effect.addFinalizer(() => Scope.close(secondScope, Exit.void));
    const secondContext = yield* buildStore(
      secondScope,
      ModelAssetStore.layerOpfs,
    );
    const secondStore = Context.get(secondContext, ModelAssetStore);

    expect(yield* secondStore.loadPackage(packageToStore.key)).toBeNull();
    expect(storage.hasPackageDirectory(packageToStore.key.packageId)).toBe(
      false,
    );
  }),
);
