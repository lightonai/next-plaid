import init, { handle_runtime_request_json } from "./pkg/next_plaid_browser_wasm.js";

const statusNode = document.getElementById("status");

function setStatus(state, value) {
  statusNode.dataset.state = state;
  statusNode.textContent =
    typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function searchRequest() {
  return {
    type: "search",
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
    query: {
      values: [
        1.0, 0.0,
        0.7, 0.7
      ],
      rows: 2,
      dim: 2
    },
    params: {
      batch_size: 2000,
      n_full_scores: 3,
      top_k: 2,
      n_ivf_probe: 2,
      centroid_batch_size: 100000,
      centroid_score_threshold: null
    },
    subset_doc_ids: null
  };
}

async function main() {
  try {
    await init();
    const responseJson = handle_runtime_request_json(
      JSON.stringify(searchRequest()),
    );
    const response = JSON.parse(responseJson);
    if (response.type !== "search_results") {
      throw new Error(`unexpected response type: ${response.type}`);
    }
    window.__NEXT_PLAID_SMOKE_RESULT__ = response;
    setStatus("ok", response);
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    window.__NEXT_PLAID_SMOKE_ERROR__ = message;
    setStatus("error", message);
  }
}

main();
