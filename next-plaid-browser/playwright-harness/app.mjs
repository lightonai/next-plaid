const statusNode = document.getElementById("status");
const WORKER_REQUEST_TIMEOUT_MS = 15_000;

function setStatus(state, value) {
  statusNode.dataset.state = state;
  statusNode.textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function loadIndexRequest() {
  return {
    type: "load_index",
    name: "demo-smoke",
    index: {
      centroids: {
        values: [
          1.0, 0.0,
          0.0, 1.0,
          0.7, 0.7
        ],
        rows: 3,
        dim: 2
      },
      ivf_doc_ids: [0, 2, 1, 2, 0, 1, 2],
      ivf_lengths: [2, 2, 3],
      doc_offsets: [0, 2, 4, 6],
      doc_codes: [0, 2, 1, 2, 2, 2],
      doc_values: [
        1.0, 0.0, 0.7, 0.7,
        0.0, 1.0, 0.7, 0.7,
        0.7, 0.7, 0.7, 0.7
      ]
    },
    metadata: [
      { title: "alpha launch memo", topic: "edge" },
      { title: "beta report summary", topic: "metrics" },
      { title: "gamma archive note", topic: "history" }
    ],
    nbits: 2,
    fts_tokenizer: "unicode61",
    max_documents: null
  };
}

async function installStoredBundleRequest() {
  const manifest = await fetch("../fixtures/demo-bundle/manifest.json").then((response) => response.json());
  const artifacts = await Promise.all(
    manifest.artifacts.map(async (artifact) => {
      const bytes = new Uint8Array(
        await fetch(`../fixtures/demo-bundle/${artifact.path}`).then((response) => response.arrayBuffer())
      );
      let binary = "";
      for (const byte of bytes) {
        binary += String.fromCharCode(byte);
      }
      return {
        kind: artifact.kind,
        bytes_b64: btoa(binary)
      };
    })
  );

  return {
    type: "install_bundle",
    manifest: {
      ...manifest,
      index_id: "demo-stored-bundle",
      build_id: "build-demo-stored-001"
    },
    artifacts,
    activate: true
  };
}

function loadStoredBundleRequest() {
  return {
    type: "load_stored_bundle",
    index_id: "demo-stored-bundle",
    name: "stored-demo",
    fts_tokenizer: "unicode61"
  };
}

function semanticSearchRequest() {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: [
        {
          embeddings: [
            [1.0, 0.0],
            [0.7, 0.7]
          ]
        }
      ],
      params: {
        top_k: 2,
        n_ivf_probe: 2,
        n_full_scores: 3,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null
    }
  };
}

function keywordSearchRequest() {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: null,
      params: {
        top_k: 2,
        n_ivf_probe: null,
        n_full_scores: null,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: ["alpha"],
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null
    }
  };
}

function hybridSearchRequest() {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: [
        {
          embeddings: [
            [0.0, 1.0],
            [0.7, 0.7]
          ]
        }
      ],
      params: {
        top_k: 2,
        n_ivf_probe: 2,
        n_full_scores: 3,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: ["beta"],
      alpha: 0.25,
      fusion: "relative_score",
      filter_condition: null,
      filter_parameters: null
    }
  };
}

function filteredSemanticSearchRequest() {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: [
        {
          embeddings: [
            [1.0, 0.0],
            [0.7, 0.7]
          ]
        }
      ],
      params: {
        top_k: 2,
        n_ivf_probe: 2,
        n_full_scores: 3,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: "topic = ?",
      filter_parameters: ["metrics"]
    }
  };
}

function filteredKeywordSearchRequest() {
  return {
    type: "search",
    name: "demo-smoke",
    request: {
      queries: null,
      params: {
        top_k: 2,
        n_ivf_probe: null,
        n_full_scores: null,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: ["alpha OR gamma"],
      alpha: null,
      fusion: null,
      filter_condition: "topic IN (?, ?)",
      filter_parameters: ["history", "edge"]
    }
  };
}

function storedKeywordSearchRequest() {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: null,
      params: {
        top_k: 2,
        n_ivf_probe: null,
        n_full_scores: null,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: ["alpha"],
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null
    }
  };
}

function storedSemanticSearchRequest() {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: [
        {
          embeddings: [
            [1.0, 0.0, 0.0, 0.0]
          ]
        }
      ],
      params: {
        top_k: 2,
        n_ivf_probe: 2,
        n_full_scores: 2,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: null,
      alpha: null,
      fusion: null,
      filter_condition: null,
      filter_parameters: null
    }
  };
}

function storedHybridSearchRequest() {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: [
        {
          embeddings: [
            [0.0, 1.0, 0.0, 0.0]
          ]
        }
      ],
      params: {
        top_k: 2,
        n_ivf_probe: 2,
        n_full_scores: 2,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: ["beta"],
      alpha: 0.25,
      fusion: "relative_score",
      filter_condition: null,
      filter_parameters: null
    }
  };
}

function storedFilteredKeywordSearchRequest() {
  return {
    type: "search",
    name: "stored-demo",
    request: {
      queries: null,
      params: {
        top_k: 2,
        n_ivf_probe: null,
        n_full_scores: null,
        centroid_score_threshold: null
      },
      subset: null,
      text_query: ["beta"],
      alpha: null,
      fusion: null,
      filter_condition: "title = ?",
      filter_parameters: ["beta"]
    }
  };
}

function callWorker(worker, request) {
  const requestId = crypto.randomUUID();

  return new Promise((resolve, reject) => {
    const timeoutId = setTimeout(() => {
      cleanup();
      reject(new Error(`Worker request timed out after ${WORKER_REQUEST_TIMEOUT_MS}ms: ${request?.type ?? "unknown"}`));
    }, WORKER_REQUEST_TIMEOUT_MS);

    const cleanup = () => {
      clearTimeout(timeoutId);
      worker.removeEventListener("message", handleMessage);
      worker.removeEventListener("error", handleError);
      worker.removeEventListener("messageerror", handleMessageError);
    };

    const handleMessage = (event) => {
      if (event.data?.requestId !== requestId) {
        return;
      }

      cleanup();
      const { ok, response, error } = event.data;
      if (ok) {
        resolve(response);
      } else {
        reject(new Error(error));
      }
    };

    const handleError = (event) => {
      cleanup();
      reject(new Error(`Worker error while handling ${request?.type ?? "unknown"}: ${event.message}`));
    };

    const handleMessageError = () => {
      cleanup();
      reject(new Error(`Worker messageerror while handling ${request?.type ?? "unknown"}`));
    };

    worker.addEventListener("message", handleMessage);
    worker.addEventListener("error", handleError);
    worker.addEventListener("messageerror", handleMessageError);
    worker.postMessage({ requestId, request });
  });
}

async function main() {
  const worker = new Worker("./worker.mjs", { type: "module" });
  let reloadWorker = null;

  try {
    const initialHealth = await callWorker(worker, { type: "health" });
    const installBundle = await callWorker(worker, await installStoredBundleRequest());
    worker.terminate();

    reloadWorker = new Worker("./worker.mjs", { type: "module" });
    const reloadedInitialHealth = await callWorker(reloadWorker, { type: "health" });
    const loadStoredBundle = await callWorker(reloadWorker, loadStoredBundleRequest());
    const storedSemanticSearch = await callWorker(reloadWorker, storedSemanticSearchRequest());
    const storedKeywordSearch = await callWorker(reloadWorker, storedKeywordSearchRequest());
    const storedHybridSearch = await callWorker(reloadWorker, storedHybridSearchRequest());
    const storedFilteredKeywordSearch = await callWorker(reloadWorker, storedFilteredKeywordSearchRequest());
    const load = await callWorker(reloadWorker, loadIndexRequest());
    const health = await callWorker(reloadWorker, { type: "health" });
    const semanticSearch = await callWorker(reloadWorker, semanticSearchRequest());
    const keywordSearch = await callWorker(reloadWorker, keywordSearchRequest());
    const hybridSearch = await callWorker(reloadWorker, hybridSearchRequest());
    const filteredSemanticSearch = await callWorker(reloadWorker, filteredSemanticSearchRequest());
    const filteredKeywordSearch = await callWorker(reloadWorker, filteredKeywordSearchRequest());

    const result = {
      initialHealth,
      installBundle,
      reloadedInitialHealth,
      loadStoredBundle,
      storedSemanticSearch,
      storedKeywordSearch,
      storedHybridSearch,
      storedFilteredKeywordSearch,
      load,
      health,
      semanticSearch,
      keywordSearch,
      hybridSearch,
      filteredSemanticSearch,
      filteredKeywordSearch
    };
    window.__NEXT_PLAID_SMOKE_RESULT__ = result;
    setStatus("ok", result);
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    window.__NEXT_PLAID_SMOKE_ERROR__ = message;
    setStatus("error", message);
  } finally {
    worker.terminate();
    reloadWorker?.terminate();
  }
}

main();
