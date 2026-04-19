import { BrowserKeywordEngine } from "../runtime/keyword_engine.mjs";
import init, {
  handle_runtime_request_json,
  reset_runtime_state
} from "./pkg/next_plaid_browser_wasm.js";

let runtimeReady;
let keywordEngine;

function runtimeCall(request) {
  const responseJson = handle_runtime_request_json(JSON.stringify(request));
  return JSON.parse(responseJson);
}

async function ensureRuntime() {
  if (!runtimeReady) {
    runtimeReady = Promise.all([init(), BrowserKeywordEngine.create()]).then(([, engine]) => {
      reset_runtime_state();
      keywordEngine = engine;
    });
  }

  await runtimeReady;
}

function hasFilterCondition(request) {
  return typeof request.filter_condition === "string" && request.filter_condition.length > 0;
}

function validateSearchRequest(request) {
  const hasQueries = Array.isArray(request.queries) && request.queries.length > 0;
  const hasTextQuery = Array.isArray(request.text_query) && request.text_query.length > 0;

  if (!hasQueries && !hasTextQuery) {
    throw new Error(
      "At least one of 'queries' (embeddings) or 'text_query' (keyword) must be provided",
    );
  }

  const alpha = request.alpha ?? 0.75;
  if (alpha < 0.0 || alpha > 1.0) {
    throw new Error("alpha must be between 0.0 and 1.0");
  }

  const fusionMode = request.fusion ?? "rrf";
  if (fusionMode !== "rrf" && fusionMode !== "relative_score") {
    throw new Error("fusion must be 'rrf' or 'relative_score'");
  }

  if (hasQueries && hasTextQuery && request.queries.length !== 1) {
    throw new Error(
      `Hybrid search requires exactly 1 query embedding (got ${request.queries.length}). text_query is a single string and can only fuse with one semantic query.`,
    );
  }

  if (hasQueries && hasTextQuery && request.queries.length !== request.text_query.length) {
    throw new Error(
      `queries length (${request.queries.length}) must match text_query length (${request.text_query.length}) in hybrid mode`,
    );
  }
}

function keywordResultsOrEmpty(name, queries, topK, subset) {
  try {
    return keywordEngine.searchIndex({ name, queries, topK, subset });
  } catch (error) {
    console.warn("[next-plaid-browser] keyword search failed", error);
    return null;
  }
}

function metadataForDocumentIds(name, documentIds) {
  return keywordEngine.metadataForDocumentIds(name, documentIds);
}

function resolveSubset(name, request) {
  if (hasFilterCondition(request)) {
    return keywordEngine.filterIndex({
      name,
      condition: request.filter_condition,
      parameters: request.filter_parameters ?? []
    });
  }

  return request.subset ?? null;
}

function buildSemanticSearchRequest(name, request, topK, subset) {
  return {
    type: "search",
    name,
    request: {
      ...request,
      subset,
      text_query: null,
      filter_condition: null,
      filter_parameters: null,
      params: {
        ...(request.params ?? {}),
        top_k: topK
      }
    }
  };
}

function buildFusionRequest(request, topK, semantic, keyword) {
  return {
    type: "fuse",
    semantic,
    keyword,
    alpha: request.alpha ?? null,
    fusion: request.fusion ?? null,
    top_k: topK
  };
}

function buildSearchResponse(results) {
  return {
    type: "search_results",
    results,
    num_queries: results.length
  };
}

function normalizeRankedResults(result) {
  if (!result) {
    return null;
  }

  return {
    document_ids: result.document_ids,
    scores: result.scores
  };
}

function emptyKeywordResults(count) {
  return Array.from({ length: count }, () => ({
    document_ids: [],
    scores: []
  }));
}

function executeSemanticSearch(name, request) {
  validateSearchRequest(request);

  const topK = request.params?.top_k ?? 10;
  const subset = resolveSubset(name, request);
  return runtimeCall(buildSemanticSearchRequest(name, request, topK, subset));
}

function executeKeywordOrHybridSearch(name, request) {
  validateSearchRequest(request);

  const hasQueries = Array.isArray(request.queries) && request.queries.length > 0;
  const textQueries = request.text_query ?? [];
  const topK = request.params?.top_k ?? 10;
  const fetchK = hasQueries ? topK * 3 : topK;
  const subset = resolveSubset(name, request);

  const semanticResponse = hasQueries
    ? runtimeCall(buildSemanticSearchRequest(name, request, fetchK, subset))
    : null;

  const keywordResponse =
    keywordResultsOrEmpty(name, textQueries, fetchK, subset) ??
    emptyKeywordResults(textQueries.length);

  const numQueries = hasQueries ? request.queries.length : textQueries.length;
  const results = [];

  for (let queryId = 0; queryId < numQueries; queryId += 1) {
    const semantic = normalizeRankedResults(semanticResponse?.results?.[queryId] ?? null);
    const keyword = normalizeRankedResults(keywordResponse[queryId] ?? null);
    const fused = runtimeCall(buildFusionRequest(request, topK, semantic, keyword));

    results.push({
      query_id: queryId,
      document_ids: fused.document_ids,
      scores: fused.scores,
      metadata: metadataForDocumentIds(name, fused.document_ids)
    });
  }

  return buildSearchResponse(results);
}

async function handleRequest(request) {
  await ensureRuntime();

  switch (request?.type) {
    case "load_index": {
      const response = runtimeCall(request);
      keywordEngine.loadIndex({
        name: request.name,
        metadata: request.metadata ?? null,
        tokenizer: request.fts_tokenizer ?? "unicode61"
      });
      return response;
    }
    case "search": {
      const hasTextQuery = Array.isArray(request.request?.text_query) && request.request.text_query.length > 0;
      const filterSearch = hasFilterCondition(request.request ?? {});
      if (!hasTextQuery && !filterSearch) {
        return runtimeCall(request);
      }
      if (!hasTextQuery) {
        return executeSemanticSearch(request.name, request.request);
      }
      return executeKeywordOrHybridSearch(request.name, request.request);
    }
    default:
      return runtimeCall(request);
  }
}

self.addEventListener("message", async (event) => {
  const { requestId, request } = event.data ?? {};

  try {
    const response = await handleRequest(request);
    self.postMessage({ requestId, ok: true, response });
  } catch (error) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    self.postMessage({ requestId, ok: false, error: message });
  }
});
