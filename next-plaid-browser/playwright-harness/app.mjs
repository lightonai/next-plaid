const statusNode = document.getElementById("status");

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
      { title: "doc-0" },
      { title: "doc-1" },
      null
    ],
    nbits: 2,
    max_documents: null
  };
}

function searchRequest() {
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

function callWorker(worker, request) {
  const requestId = crypto.randomUUID();

  return new Promise((resolve, reject) => {
    const handleMessage = (event) => {
      if (event.data?.requestId !== requestId) {
        return;
      }

      worker.removeEventListener("message", handleMessage);
      const { ok, response, error } = event.data;
      if (ok) {
        resolve(response);
      } else {
        reject(new Error(error));
      }
    };

    worker.addEventListener("message", handleMessage);
    worker.postMessage({ requestId, request });
  });
}

async function main() {
  const worker = new Worker("./worker.mjs", { type: "module" });

  try {
    const initialHealth = await callWorker(worker, { type: "health" });
    const load = await callWorker(worker, loadIndexRequest());
    const health = await callWorker(worker, { type: "health" });
    const search = await callWorker(worker, searchRequest());

    const result = { initialHealth, load, health, search };
    window.__NEXT_PLAID_SMOKE_RESULT__ = result;
    setStatus("ok", result);
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    window.__NEXT_PLAID_SMOKE_ERROR__ = message;
    setStatus("error", message);
  } finally {
    worker.terminate();
  }
}

main();
