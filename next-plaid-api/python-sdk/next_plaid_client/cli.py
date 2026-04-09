"""
CLI for the Next Plaid ColBERT Search API.

Uses the Python SDK to provide a non-interactive, agent-friendly CLI
following resource-verb patterns with layered --help and examples.
"""

import json
import sys

import click

from .client import NextPlaidClient
from .exceptions import NextPlaidError
from .models import IndexConfig, SearchParams


def _get_client(ctx: click.Context) -> NextPlaidClient:
    """Build a client from the global options stored in ctx.obj."""
    return NextPlaidClient(
        base_url=ctx.obj["url"],
        timeout=ctx.obj["timeout"],
        headers=ctx.obj.get("headers"),
    )


def _output(ctx: click.Context, data, human_fn=None):
    """Print output as JSON or human-readable text."""
    if human_fn and not ctx.obj.get("json"):
        click.echo(human_fn(data))
    else:
        serializable = _to_dict(data) if hasattr(data, "__dict__") else data
        click.echo(json.dumps(serializable, indent=2))


def _to_dict(obj):
    """Recursively convert dataclass-like objects to dicts."""
    if hasattr(obj, "__dict__"):
        return {
            k: _to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")
        }
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _read_stdin_or_fail(what: str) -> str:
    """Read stdin if available, otherwise fail with guidance."""
    if sys.stdin.isatty():
        raise click.UsageError(
            f"No {what} provided. Pass via --stdin or as arguments.\n"
            f"  Example: echo '...' | next-plaid ..."
        )
    return sys.stdin.read()


def _parse_params(params: tuple) -> list | None:
    """Parse CLI string params into typed values (int > float > str)."""
    if not params:
        return None
    result = []
    for p in params:
        try:
            result.append(int(p))
        except ValueError:
            try:
                result.append(float(p))
            except ValueError:
                result.append(p)
    return result


def _parse_json_param(value: str, name: str):
    """Parse a JSON string parameter."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise click.BadParameter(f"Invalid JSON for {name}: {e}")


def _handle_error(e: NextPlaidError):
    """Print a structured error and exit."""
    msg = f"Error: {e.message}"
    if e.code:
        msg += f" [{e.code}]"
    if e.details:
        msg += f"\n  details: {e.details}"
    click.echo(msg, err=True)
    sys.exit(1)


# ──────────────────────────── Root group ────────────────────────────


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="Run 'next-plaid <command> --help' for subcommand details.",
)
@click.option(
    "--url",
    "-u",
    envvar="NEXT_PLAID_URL",
    default="http://localhost:8080",
    show_default=True,
    help="Server URL (env: NEXT_PLAID_URL).",
)
@click.option(
    "--timeout",
    "-t",
    type=float,
    default=30.0,
    show_default=True,
    help="Request timeout in seconds.",
)
@click.option(
    "--header", "-H", multiple=True, help="Extra header as 'Key: Value'. Repeatable."
)
@click.option("--json", "json_output", is_flag=True, help="Output full JSON responses.")
@click.version_option(package_name="next-plaid-client")
@click.pass_context
def cli(ctx, url, timeout, header, json_output):
    """Next Plaid CLI - manage indices, documents, and search."""
    headers = {}
    for h in header:
        if ":" not in h:
            raise click.BadParameter(f"Header must be 'Key: Value', got: {h}")
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    ctx.ensure_object(dict)
    ctx.obj["url"] = url
    ctx.obj["timeout"] = timeout
    ctx.obj["headers"] = headers or None
    ctx.obj["json"] = json_output


# ──────────────────────────── health ────────────────────────────


@cli.command()
@click.pass_context
def health(ctx):
    """Check server health and status.

    \b
    Examples:
      next-plaid health
      next-plaid health --json
      next-plaid -u http://remote:8080 health
    """
    try:
        with _get_client(ctx) as client:
            h = client.health()
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(h):
        lines = [
            f"status: {h.status}",
            f"version: {h.version}",
            f"loaded_indices: {h.loaded_indices}",
            f"index_dir: {h.index_dir}",
            f"memory_bytes: {h.memory_usage_bytes}",
        ]
        if h.indices:
            lines.append("indices:")
            for idx in h.indices:
                lines.append(
                    f"  - {idx.name}  docs={idx.num_documents}  dim={idx.dimension}  nbits={idx.nbits}"
                )
        return "\n".join(lines)

    _output(ctx, h, _fmt)


# ──────────────────────────── index ────────────────────────────


@cli.group()
def index():
    """Manage indices.

    \b
    Examples:
      next-plaid index list
      next-plaid index create my_index --nbits 4
      next-plaid index get my_index
      next-plaid index delete my_index --yes
    """


@index.command("list")
@click.pass_context
def index_list(ctx):
    """List all index names.

    \b
    Examples:
      next-plaid index list
      next-plaid index list --json
    """
    try:
        with _get_client(ctx) as client:
            names = client.list_indices()
    except NextPlaidError as e:
        _handle_error(e)

    _output(ctx, names, lambda ns: "\n".join(ns) if ns else "(no indices)")


@index.command("get")
@click.argument("name")
@click.pass_context
def index_get(ctx, name):
    """Get detailed info for an index.

    \b
    Examples:
      next-plaid index get my_index
      next-plaid index get my_index --json
    """
    try:
        with _get_client(ctx) as client:
            info = client.get_index(name)
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(i):
        lines = [
            f"name: {i.name}",
            f"documents: {i.num_documents}",
            f"embeddings: {i.num_embeddings}",
            f"partitions: {i.num_partitions}",
            f"avg_doclen: {i.avg_doclen:.1f}",
            f"dimension: {i.dimension}",
            f"has_metadata: {i.has_metadata}",
        ]
        if i.metadata_count is not None:
            lines.append(f"metadata_count: {i.metadata_count}")
        if i.max_documents is not None:
            lines.append(f"max_documents: {i.max_documents}")
        return "\n".join(lines)

    _output(ctx, info, _fmt)


@index.command("create")
@click.argument("name")
@click.option(
    "--nbits",
    type=click.Choice(["2", "4"]),
    default="4",
    show_default=True,
    help="Quantization bits.",
)
@click.option(
    "--batch-size",
    type=int,
    default=50000,
    show_default=True,
    help="Documents per indexing batch.",
)
@click.option("--seed", type=int, default=None, help="Random seed for K-means.")
@click.option(
    "--max-documents",
    type=int,
    default=None,
    help="Max documents (evicts oldest when exceeded).",
)
@click.option(
    "--fts-tokenizer",
    type=click.Choice(["unicode61", "trigram"]),
    default=None,
    help="FTS5 tokenizer.",
)
@click.pass_context
def index_create(ctx, name, nbits, batch_size, seed, max_documents, fts_tokenizer):
    """Create a new index.

    \b
    Examples:
      next-plaid index create my_index
      next-plaid index create my_index --nbits 2 --max-documents 10000
      next-plaid index create code_index --fts-tokenizer trigram
    """
    config = IndexConfig(
        nbits=int(nbits),
        batch_size=batch_size,
        seed=seed,
        max_documents=max_documents,
        fts_tokenizer=fts_tokenizer,
    )
    try:
        with _get_client(ctx) as client:
            result = client.create_index(name, config)
    except NextPlaidError as e:
        _handle_error(e)

    _output(ctx, result, lambda r: f"created index: {name}")


@index.command("delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be deleted without acting."
)
@click.pass_context
def index_delete(ctx, name, yes, dry_run):
    """Delete an index and all its data.

    This is destructive and cannot be undone.

    \b
    Examples:
      next-plaid index delete my_index --yes
      next-plaid index delete my_index --dry-run
    """
    if dry_run:
        click.echo(f"dry-run: would delete index '{name}' and all its data")
        return

    if not yes:
        click.confirm(f"Delete index '{name}' and all its data?", abort=True)

    try:
        with _get_client(ctx) as client:
            result = client.delete_index(name)
    except NextPlaidError as e:
        _handle_error(e)

    _output(ctx, result, lambda r: f"deleted index: {name}")


@index.command("config")
@click.argument("name")
@click.option(
    "--max-documents",
    type=int,
    default=None,
    help="Set max documents limit (0 to remove).",
)
@click.pass_context
def index_config(ctx, name, max_documents):
    """Update index configuration.

    \b
    Examples:
      next-plaid index config my_index --max-documents 10000
      next-plaid index config my_index --max-documents 0
    """
    if max_documents == 0:
        max_documents = None

    try:
        with _get_client(ctx) as client:
            result = client.update_index_config(name, max_documents)
    except NextPlaidError as e:
        _handle_error(e)

    _output(ctx, result, lambda r: f"updated config for index: {name}")


# ──────────────────────────── document ────────────────────────────


@cli.group()
def document():
    """Add or delete documents.

    \b
    Examples:
      next-plaid document add my_index --text "Hello world" --text "Another doc"
      next-plaid document add my_index --file docs.json
      cat texts.jsonl | next-plaid document add my_index --stdin
      next-plaid document delete my_index --condition "category = ?" --param draft
    """


@document.command("add")
@click.argument("index_name")
@click.option("--text", "-t", multiple=True, help="Text document to add. Repeatable.")
@click.option(
    "--file",
    "-f",
    "filepath",
    type=click.Path(exists=True),
    help="JSON file with documents.",
)
@click.option(
    "--stdin",
    "use_stdin",
    is_flag=True,
    help="Read documents from stdin (JSON array of strings or objects).",
)
@click.option(
    "--metadata-file",
    type=click.Path(exists=True),
    help="JSON file with metadata array.",
)
@click.option(
    "--pool-factor",
    type=int,
    default=None,
    help="Token pooling factor (e.g., 2 for 2x reduction).",
)
@click.pass_context
def document_add(
    ctx, index_name, text, filepath, use_stdin, metadata_file, pool_factor
):
    """Add documents to an index.

    Accepts text strings (server encodes) or pre-computed embeddings as JSON.

    \b
    Text input:
      next-plaid document add my_index --text "Paris is the capital of France"
      next-plaid document add my_index --text "Doc 1" --text "Doc 2"

    \b
    JSON file with text array:
      next-plaid document add my_index --file texts.json
      # texts.json: ["Document one", "Document two"]

    \b
    JSON file with embeddings:
      next-plaid document add my_index --file embeddings.json
      # embeddings.json: [{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}]

    \b
    Stdin (pipe text array):
      echo '["Doc one", "Doc two"]' | next-plaid document add my_index --stdin

    \b
    With metadata:
      next-plaid document add my_index --text "Paris" --metadata-file meta.json
      # meta.json: [{"country": "France"}]
    """
    documents = None
    metadata = None

    if text:
        documents = list(text)
    elif filepath:
        with open(filepath) as f:
            documents = json.load(f)
    elif use_stdin:
        raw = _read_stdin_or_fail("documents")
        documents = json.loads(raw)
    else:
        raise click.UsageError(
            "No documents provided. Use --text, --file, or --stdin.\n"
            '  next-plaid document add my_index --text "Hello world"\n'
            "  next-plaid document add my_index --file docs.json\n"
            "  echo '[\"text\"]' | next-plaid document add my_index --stdin"
        )

    if metadata_file:
        with open(metadata_file) as f:
            metadata = json.load(f)

    try:
        with _get_client(ctx) as client:
            result = client.add(
                index_name, documents, metadata=metadata, pool_factor=pool_factor
            )
    except NextPlaidError as e:
        _handle_error(e)

    _output(
        ctx,
        {"status": result, "index": index_name, "count": len(documents)},
        lambda r: f"queued {r['count']} document(s) for index: {r['index']}",
    )


@document.command("delete")
@click.argument("index_name")
@click.option(
    "--condition",
    "-c",
    required=True,
    help='SQL WHERE condition (e.g., "category = ?").',
)
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="Parameter for condition placeholder. Repeatable, in order.",
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.option(
    "--dry-run", is_flag=True, help="Show the delete condition without executing."
)
@click.pass_context
def document_delete(ctx, index_name, condition, param, yes, dry_run):
    """Delete documents matching a metadata condition.

    \b
    Examples:
      next-plaid document delete my_index --condition "category = ?" --param draft --yes
      next-plaid document delete my_index --condition "year < ?" --param 2020 --yes
      next-plaid document delete my_index --condition "id IN (?, ?)" --param 1 --param 2 --dry-run
    """
    parameters = _parse_params(param)

    if dry_run:
        click.echo(f"dry-run: would delete from '{index_name}' where {condition}")
        if parameters:
            click.echo(f"  parameters: {parameters}")
        return

    if not yes:
        click.confirm(
            f"Delete documents from '{index_name}' where {condition}?", abort=True
        )

    try:
        with _get_client(ctx) as client:
            result = client.delete(index_name, condition, parameters=parameters)
    except NextPlaidError as e:
        _handle_error(e)

    _output(
        ctx,
        {"status": result, "index": index_name},
        lambda r: f"delete queued for index: {r['index']}",
    )


# ──────────────────────────── search ────────────────────────────


@cli.command()
@click.argument("index_name")
@click.argument("queries", nargs=-1)
@click.option(
    "--stdin",
    "use_stdin",
    is_flag=True,
    help="Read queries from stdin (JSON array of strings).",
)
@click.option(
    "--top-k", "-k", type=int, default=10, show_default=True, help="Results per query."
)
@click.option(
    "--n-ivf-probe", type=int, default=None, help="IVF cells to probe (default: 8)."
)
@click.option(
    "--n-full-scores",
    type=int,
    default=None,
    help="Candidates for re-ranking (default: 4096).",
)
@click.option(
    "--centroid-threshold",
    type=float,
    default=None,
    help="Centroid score threshold (default: 0.4, 0 to disable).",
)
@click.option(
    "--filter", "filter_condition", default=None, help="SQL WHERE filter on metadata."
)
@click.option(
    "--filter-param",
    multiple=True,
    help="Parameter for filter placeholder. Repeatable.",
)
@click.option(
    "--text-query",
    multiple=True,
    help="FTS5 keyword query for hybrid/keyword search. Repeatable.",
)
@click.option(
    "--alpha",
    type=float,
    default=None,
    help="Hybrid balance: 0=keyword, 1=semantic (default: 0.75).",
)
@click.option(
    "--fusion",
    type=click.Choice(["rrf", "relative_score"]),
    default=None,
    help="Hybrid fusion strategy.",
)
@click.option(
    "--subset",
    multiple=True,
    type=int,
    help="Restrict search to these document IDs. Repeatable.",
)
@click.pass_context
def search(
    ctx,
    index_name,
    queries,
    use_stdin,
    top_k,
    n_ivf_probe,
    n_full_scores,
    centroid_threshold,
    filter_condition,
    filter_param,
    text_query,
    alpha,
    fusion,
    subset,
):
    """Search an index with text queries, keyword queries, or both (hybrid).

    QUERIES are text strings to search semantically. Use --text-query for
    keyword (FTS5) search. Combine both for hybrid search.

    \b
    Semantic search:
      next-plaid search my_index "What is machine learning?"
      next-plaid search my_index "query one" "query two" --top-k 5

    \b
    Keyword search:
      next-plaid search my_index --text-query "machine learning"

    \b
    Hybrid search:
      next-plaid search my_index "What is ML?" --text-query "machine learning"

    \b
    With metadata filter:
      next-plaid search my_index "AI papers" --filter "year > ?" --filter-param 2022

    \b
    Restrict to document subset:
      next-plaid search my_index "query" --subset 1 --subset 2 --subset 3

    \b
    From stdin:
      echo '["query 1", "query 2"]' | next-plaid search my_index --stdin
    """
    subset_list = list(subset) if subset else None

    query_list = None
    if queries:
        query_list = list(queries)
    elif use_stdin:
        raw = _read_stdin_or_fail("queries")
        query_list = json.loads(raw)

    text_query_list = list(text_query) if text_query else None

    if not query_list and not text_query_list:
        raise click.UsageError(
            "No queries provided. Pass text queries as arguments or use --text-query.\n"
            '  next-plaid search my_index "What is AI?"\n'
            '  next-plaid search my_index --text-query "artificial intelligence"'
        )

    params = SearchParams(top_k=top_k)
    if n_ivf_probe is not None:
        params.n_ivf_probe = n_ivf_probe
    if n_full_scores is not None:
        params.n_full_scores = n_full_scores
    if centroid_threshold is not None:
        params.centroid_score_threshold = (
            centroid_threshold if centroid_threshold > 0 else None
        )

    filter_parameters = _parse_params(filter_param)

    try:
        with _get_client(ctx) as client:
            if text_query_list and not query_list:
                result = client.keyword_search(
                    index_name,
                    text_query_list,
                    params=params,
                    filter_condition=filter_condition,
                    filter_parameters=filter_parameters,
                )
            else:
                result = client.search(
                    index_name,
                    query_list,
                    params=params,
                    filter_condition=filter_condition,
                    filter_parameters=filter_parameters,
                    subset=subset_list,
                    text_query=text_query_list,
                    alpha=alpha,
                    fusion=fusion,
                )
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(r):
        lines = [f"num_queries: {r.num_queries}"]
        for qr in r.results:
            lines.append(f"query {qr.query_id}:")
            for doc_id, score in zip(qr.document_ids, qr.scores):
                line = f"  doc_id={doc_id}  score={score:.4f}"
                lines.append(line)
            if qr.metadata:
                for i, m in enumerate(qr.metadata):
                    if m:
                        lines.append(f"  metadata[{i}]: {json.dumps(m)}")
        return "\n".join(lines)

    _output(ctx, result, _fmt)


# ──────────────────────────── metadata ────────────────────────────


@cli.group()
def metadata():
    """Query and manage document metadata.

    \b
    Examples:
      next-plaid metadata count my_index
      next-plaid metadata list my_index
      next-plaid metadata query my_index --condition "category = ?" --param science
      next-plaid metadata get my_index --ids 1 --ids 2
    """


@metadata.command("list")
@click.argument("index_name")
@click.pass_context
def metadata_list(ctx, index_name):
    """Get all metadata entries for an index.

    \b
    Examples:
      next-plaid metadata list my_index
      next-plaid metadata list my_index --json
    """
    try:
        with _get_client(ctx) as client:
            result = client.get_metadata(index_name)
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(r):
        lines = [f"count: {r.count}"]
        for m in r.metadata:
            lines.append(f"  {json.dumps(m)}")
        return "\n".join(lines)

    _output(ctx, result, _fmt)


@metadata.command("count")
@click.argument("index_name")
@click.pass_context
def metadata_count(ctx, index_name):
    """Count metadata entries.

    \b
    Examples:
      next-plaid metadata count my_index
    """
    try:
        with _get_client(ctx) as client:
            result = client.get_metadata_count(index_name)
    except NextPlaidError as e:
        _handle_error(e)

    _output(
        ctx,
        result,
        lambda r: (
            f"count: {r.get('count', 0)}\nhas_metadata: {r.get('has_metadata', False)}"
        ),
    )


@metadata.command("check")
@click.argument("index_name")
@click.option(
    "--ids",
    "-i",
    multiple=True,
    type=int,
    required=True,
    help="Document ID to check. Repeatable.",
)
@click.pass_context
def metadata_check(ctx, index_name, ids):
    """Check which document IDs have metadata.

    \b
    Examples:
      next-plaid metadata check my_index --ids 1 --ids 2 --ids 3
    """
    try:
        with _get_client(ctx) as client:
            result = client.check_metadata(index_name, list(ids))
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(r):
        return (
            f"existing: {r.existing_ids} ({r.existing_count})\n"
            f"missing: {r.missing_ids} ({r.missing_count})"
        )

    _output(ctx, result, _fmt)


@metadata.command("query")
@click.argument("index_name")
@click.option("--condition", "-c", required=True, help="SQL WHERE condition.")
@click.option("--param", "-p", multiple=True, help="Condition parameter. Repeatable.")
@click.pass_context
def metadata_query(ctx, index_name, condition, param):
    """Query metadata by SQL condition, returning matching document IDs.

    \b
    Examples:
      next-plaid metadata query my_index --condition "category = ?" --param science
      next-plaid metadata query my_index --condition "score > ?" --param 90
    """
    parameters = _parse_params(param)

    try:
        with _get_client(ctx) as client:
            result = client.query_metadata(index_name, condition, parameters=parameters)
    except NextPlaidError as e:
        _handle_error(e)

    _output(
        ctx,
        result,
        lambda r: (
            f"document_ids: {r.get('document_ids', [])}\ncount: {r.get('count', 0)}"
        ),
    )


@metadata.command("get")
@click.argument("index_name")
@click.option("--ids", "-i", multiple=True, type=int, help="Document ID. Repeatable.")
@click.option("--condition", "-c", default=None, help="SQL WHERE condition.")
@click.option("--param", "-p", multiple=True, help="Condition parameter. Repeatable.")
@click.option("--limit", "-l", type=int, default=None, help="Max results.")
@click.pass_context
def metadata_get(ctx, index_name, ids, condition, param, limit):
    """Get metadata by document IDs or SQL condition.

    \b
    Examples:
      next-plaid metadata get my_index --ids 1 --ids 2
      next-plaid metadata get my_index --condition "category = ?" --param science
      next-plaid metadata get my_index --condition "score > ?" --param 90 --limit 10
    """
    parameters = _parse_params(param)

    doc_ids = list(ids) if ids else None

    try:
        with _get_client(ctx) as client:
            result = client.get_metadata_by_ids(
                index_name,
                document_ids=doc_ids,
                condition=condition,
                parameters=parameters,
                limit=limit,
            )
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(r):
        lines = [f"count: {r.count}"]
        for m in r.metadata:
            lines.append(f"  {json.dumps(m)}")
        return "\n".join(lines)

    _output(ctx, result, _fmt)


@metadata.command("update")
@click.argument("index_name")
@click.option(
    "--condition", "-c", required=True, help="SQL WHERE condition for rows to update."
)
@click.option("--param", "-p", multiple=True, help="Condition parameter. Repeatable.")
@click.option(
    "--set",
    "updates_json",
    required=True,
    help='JSON object with column updates (e.g., \'{"status": "done"}\').',
)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation.")
@click.option(
    "--dry-run", is_flag=True, help="Show what would be updated without acting."
)
@click.pass_context
def metadata_update(ctx, index_name, condition, param, updates_json, yes, dry_run):
    """Update metadata rows matching a condition.

    \b
    Examples:
      next-plaid metadata update my_index -c "status = ?" -p draft --set '{"status": "published"}' --yes
      next-plaid metadata update my_index -c "score > ?" -p 90 --set '{"reviewed": true}' --dry-run
    """
    updates = _parse_json_param(updates_json, "--set")
    parameters = _parse_params(param)

    if dry_run:
        click.echo(f"dry-run: would update '{index_name}' where {condition}")
        if parameters:
            click.echo(f"  parameters: {parameters}")
        click.echo(f"  set: {json.dumps(updates)}")
        return

    if not yes:
        click.confirm(
            f"Update metadata in '{index_name}' where {condition}?", abort=True
        )

    try:
        with _get_client(ctx) as client:
            result = client.update_metadata(
                index_name, condition, updates, parameters=parameters
            )
    except NextPlaidError as e:
        _handle_error(e)

    _output(ctx, result, lambda r: f"updated: {r.get('updated', 0)} row(s)")


# ──────────────────────────── encode ────────────────────────────


@cli.command()
@click.argument("texts", nargs=-1)
@click.option(
    "--stdin", "use_stdin", is_flag=True, help="Read texts from stdin (JSON array)."
)
@click.option(
    "--input-type",
    type=click.Choice(["document", "query"]),
    default="document",
    show_default=True,
    help="Encoding mode.",
)
@click.option(
    "--pool-factor", type=int, default=None, help="Token pooling reduction factor."
)
@click.pass_context
def encode(ctx, texts, use_stdin, input_type, pool_factor):
    """Encode texts into ColBERT embeddings (requires model on server).

    \b
    Examples:
      next-plaid encode "Hello world" "Another text"
      next-plaid encode --input-type query "What is AI?"
      echo '["text1", "text2"]' | next-plaid encode --stdin
    """
    text_list = None
    if texts:
        text_list = list(texts)
    elif use_stdin:
        raw = _read_stdin_or_fail("texts")
        text_list = json.loads(raw)
    else:
        raise click.UsageError(
            "No texts provided. Pass as arguments or use --stdin.\n"
            '  next-plaid encode "Hello world"\n'
            "  echo '[\"text\"]' | next-plaid encode --stdin"
        )

    try:
        with _get_client(ctx) as client:
            result = client.encode(
                text_list, input_type=input_type, pool_factor=pool_factor
            )
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(r):
        lines = [f"num_texts: {r.num_texts}"]
        for i, emb in enumerate(r.embeddings):
            lines.append(
                f"  text[{i}]: {len(emb)} tokens x {len(emb[0]) if emb else 0} dims"
            )
        return "\n".join(lines)

    _output(ctx, result, _fmt)


# ──────────────────────────── rerank ────────────────────────────


@cli.command()
@click.option("--query", "-q", required=True, help="Query text for reranking.")
@click.option(
    "--document",
    "-d",
    "documents",
    multiple=True,
    help="Document text to rerank. Repeatable.",
)
@click.option(
    "--file",
    "-f",
    "filepath",
    type=click.Path(exists=True),
    help="JSON file with documents array.",
)
@click.option(
    "--stdin",
    "use_stdin",
    is_flag=True,
    help="Read documents from stdin (JSON array of strings).",
)
@click.option(
    "--pool-factor", type=int, default=None, help="Token pooling reduction factor."
)
@click.pass_context
def rerank(ctx, query, documents, filepath, use_stdin, pool_factor):
    """Rerank documents by relevance to a query using ColBERT MaxSim.

    \b
    Examples:
      next-plaid rerank -q "capital of France" -d "Paris is in France" -d "Berlin is in Germany"
      next-plaid rerank -q "machine learning" --file docs.json
      echo '["doc1", "doc2"]' | next-plaid rerank -q "my query" --stdin
    """
    doc_list = None
    if documents:
        doc_list = list(documents)
    elif filepath:
        with open(filepath) as f:
            doc_list = json.load(f)
    elif use_stdin:
        raw = _read_stdin_or_fail("documents")
        doc_list = json.loads(raw)
    else:
        raise click.UsageError(
            "No documents provided. Use --document, --file, or --stdin.\n"
            '  next-plaid rerank -q "query" -d "doc one" -d "doc two"\n'
            '  next-plaid rerank -q "query" --file docs.json'
        )

    try:
        with _get_client(ctx) as client:
            result = client.rerank(query, doc_list, pool_factor=pool_factor)
    except NextPlaidError as e:
        _handle_error(e)

    def _fmt(r):
        lines = [f"num_documents: {r.num_documents}"]
        for rr in r.results:
            lines.append(f"  [{rr.index}] score={rr.score:.4f}")
        return "\n".join(lines)

    _output(ctx, result, _fmt)


def main():
    cli()


if __name__ == "__main__":
    main()
