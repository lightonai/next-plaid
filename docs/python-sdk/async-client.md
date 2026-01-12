# Async Client

`AsyncNextPlaidClient` provides asynchronous access to the NextPlaid API using `async`/`await`.

## Initialization

```python
from next_plaid_client import AsyncNextPlaidClient

client = AsyncNextPlaidClient(
    base_url="http://localhost:8080",
    timeout=30.0,
    headers=None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://localhost:8080"` | API server URL |
| `timeout` | `float` | `30.0` | Request timeout in seconds |
| `headers` | `Dict[str, str]` | `None` | Custom headers for all requests |

## Async Context Manager

```python
async with AsyncNextPlaidClient("http://localhost:8080") as client:
    health = await client.health()
    # Client is automatically closed when exiting the block
```

---

## Usage Example

```python
import asyncio
from next_plaid_client import AsyncNextPlaidClient, IndexConfig, SearchParams

async def main():
    async with AsyncNextPlaidClient("http://localhost:8080") as client:
        # Check health
        health = await client.health()
        print(f"Status: {health.status}")

        # Create index
        await client.create_index("async_index", IndexConfig(nbits=4))

        # Add documents
        documents = [{"embeddings": [[0.1, 0.2], [0.3, 0.4]]}]
        await client.add_documents("async_index", documents)

        # Search
        results = await client.search(
            "async_index",
            queries=[[[0.1, 0.2]]],
            params=SearchParams(top_k=5)
        )
        print(f"Found: {results.num_queries} queries processed")

asyncio.run(main())
```

---

## Concurrent Operations

The async client excels at concurrent operations:

```python
async def search_multiple_indices(client, queries):
    """Search multiple indices concurrently."""
    tasks = [
        client.search(f"index_{i}", queries)
        for i in range(10)
    ]
    results = await asyncio.gather(*tasks)
    return results

async def batch_add_documents(client, index_name, document_batches):
    """Add multiple document batches concurrently."""
    tasks = [
        client.add_documents(index_name, batch)
        for batch in document_batches
    ]
    await asyncio.gather(*tasks)
```

---

## API Reference

The async client has the same methods as the sync client, but all methods are `async`:

### Health & Monitoring

```python
health = await client.health()
```

### Index Management

```python
indices = await client.list_indices()
info = await client.get_index("my_index")
await client.create_index("my_index", IndexConfig())
await client.delete_index("my_index")
await client.update_index_config("my_index", max_documents=1000)
```

### Document Management

```python
await client.add_documents("my_index", documents, metadata)
await client.update_documents("my_index", documents, metadata)
await client.update_documents_with_encoding("my_index", texts, metadata)
result = await client.delete_documents("my_index", [0, 1, 2])
```

### Search Operations

```python
results = await client.search("my_index", queries, params)
results = await client.search_filtered("my_index", queries, condition, params)
results = await client.search_with_encoding("my_index", text_queries, params)
results = await client.search_filtered_with_encoding("my_index", text_queries, condition)
```

### Metadata Management

```python
response = await client.get_metadata("my_index")
await client.add_metadata("my_index", metadata)
result = await client.query_metadata("my_index", condition, parameters)
response = await client.get_metadata_by_ids("my_index", document_ids)
count = await client.get_metadata_count("my_index")
check = await client.check_metadata("my_index", document_ids)
```

### Text Encoding

```python
response = await client.encode(texts, input_type="document")
```

### Cleanup

```python
await client.close()
```

---

## Integration with Web Frameworks

### FastAPI

```python
from fastapi import FastAPI, Depends
from next_plaid_client import AsyncNextPlaidClient, SearchParams

app = FastAPI()
client = AsyncNextPlaidClient("http://localhost:8080")

@app.on_event("shutdown")
async def shutdown():
    await client.close()

@app.post("/search")
async def search(query: str):
    # First encode the query (if model available)
    results = await client.search_with_encoding(
        "my_index",
        queries=[query],
        params=SearchParams(top_k=10)
    )
    return results.results[0].document_ids
```

### aiohttp

```python
from aiohttp import web
from next_plaid_client import AsyncNextPlaidClient

async def init_app():
    app = web.Application()
    app["plaid_client"] = AsyncNextPlaidClient("http://localhost:8080")

    async def cleanup(app):
        await app["plaid_client"].close()

    app.on_cleanup.append(cleanup)
    return app

async def search_handler(request):
    client = request.app["plaid_client"]
    results = await client.search("my_index", queries=[...])
    return web.json_response({"results": results.results})
```

---

## Error Handling

```python
from next_plaid_client import (
    AsyncNextPlaidClient,
    IndexNotFoundError,
    RateLimitError,
)

async def safe_search(client, index_name, queries):
    try:
        return await client.search(index_name, queries)
    except IndexNotFoundError:
        print(f"Index {index_name} not found")
        return None
    except RateLimitError:
        print("Rate limited, retrying...")
        await asyncio.sleep(1)
        return await safe_search(client, index_name, queries)
```

---

## Complete Example

```python
import asyncio
from next_plaid_client import (
    AsyncNextPlaidClient,
    IndexConfig,
    SearchParams,
)

async def main():
    async with AsyncNextPlaidClient("http://localhost:8080") as client:
        # Setup
        indices = await client.list_indices()
        if "demo" not in indices:
            await client.create_index("demo", IndexConfig(nbits=4))

        # Add documents with text encoding
        await client.update_documents_with_encoding(
            "demo",
            documents=[
                "The Eiffel Tower is in Paris, France.",
                "The Great Wall is in China.",
                "The Colosseum is in Rome, Italy.",
            ],
            metadata=[
                {"landmark": "Eiffel Tower", "country": "France"},
                {"landmark": "Great Wall", "country": "China"},
                {"landmark": "Colosseum", "country": "Italy"},
            ]
        )

        # Search
        results = await client.search_with_encoding(
            "demo",
            queries=["Famous landmarks in Europe"],
            params=SearchParams(top_k=3)
        )

        # Get metadata for results
        doc_ids = results.results[0].document_ids
        metadata = await client.get_metadata_by_ids("demo", doc_ids)

        for doc_id, score, meta in zip(
            results.results[0].document_ids,
            results.results[0].scores,
            metadata.metadata
        ):
            print(f"{meta['landmark']}: {score:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```
