# Metadata API

Endpoints for managing document metadata. Metadata is stored in SQLite and enables filtered search.

## Get All Metadata

<span class="api-method get">GET</span> `/indices/{name}/metadata`

Get all metadata entries for an index.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

=== "Request"

    ```bash
    curl http://localhost:8080/indices/my_index/metadata
    ```

=== "Response"

    ```json
    {
      "metadata": [
        {"_id": 0, "title": "Document 1", "category": "science"},
        {"_id": 1, "title": "Document 2", "category": "history"},
        {"_id": 2, "title": "Document 3", "category": "science"}
      ],
      "count": 3
    }
    ```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | `array` | List of metadata entries |
| `count` | `int` | Total number of entries |

!!! note "Reserved Field"
    The `_id` field is automatically added and corresponds to the document ID.

---

## Add Metadata

<span class="api-method post">POST</span> `/indices/{name}/metadata`

Add or update metadata entries.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `metadata` | `array` | Yes | List of metadata objects |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/metadata \
      -H "Content-Type: application/json" \
      -d '{
        "metadata": [
          {"_id": 0, "title": "Updated Title", "category": "science"},
          {"_id": 1, "title": "New Document", "year": 2024}
        ]
      }'
    ```

=== "Response"

    ```json
    {
      "added": 2
    }
    ```

### Notes

- Include `_id` to update existing metadata
- Omit `_id` when adding new metadata (auto-assigned based on document order)
- Metadata is typically added with documents via `add_documents`

---

## Get Metadata Count

<span class="api-method get">GET</span> `/indices/{name}/metadata/count`

Get the count of metadata entries.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

=== "Request"

    ```bash
    curl http://localhost:8080/indices/my_index/metadata/count
    ```

=== "Response"

    ```json
    {
      "count": 1000,
      "has_metadata": true
    }
    ```

---

## Check Metadata Exists

<span class="api-method post">POST</span> `/indices/{name}/metadata/check`

Check which document IDs have metadata entries.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_ids` | `array` | Yes | List of document IDs to check |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/metadata/check \
      -H "Content-Type: application/json" \
      -d '{"document_ids": [0, 1, 2, 999, 1000]}'
    ```

=== "Response"

    ```json
    {
      "existing_ids": [0, 1, 2],
      "missing_ids": [999, 1000],
      "existing_count": 3,
      "missing_count": 2
    }
    ```

---

## Query Metadata

<span class="api-method post">POST</span> `/indices/{name}/metadata/query`

Query metadata using SQL WHERE conditions. Returns matching document IDs.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `condition` | `string` | Yes | SQL WHERE condition |
| `parameters` | `array` | No | Parameters for placeholders |

=== "Request"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/metadata/query \
      -H "Content-Type: application/json" \
      -d '{
        "condition": "category = ? AND year >= ?",
        "parameters": ["science", 2020]
      }'
    ```

=== "Response"

    ```json
    {
      "document_ids": [0, 2, 5, 8, 12],
      "count": 5
    }
    ```

### Filter Syntax

The condition uses SQLite syntax:

```sql
-- Simple equality
category = ?

-- Multiple conditions
category = ? AND year >= ?

-- OR conditions
category = ? OR category = ?

-- IN clause (use multiple ? placeholders)
category IN (?, ?, ?)

-- LIKE for pattern matching
title LIKE ?  -- Use "%" as wildcard in parameter

-- NULL checks
description IS NOT NULL

-- Comparison operators
year > ? AND year < ?
score >= ?
```

### Example Queries

**Category filter:**

```json
{
  "condition": "category = ?",
  "parameters": ["science"]
}
```

**Year range:**

```json
{
  "condition": "year >= ? AND year <= ?",
  "parameters": [2020, 2024]
}
```

**Text search:**

```json
{
  "condition": "title LIKE ?",
  "parameters": ["%machine learning%"]
}
```

**Multiple categories:**

```json
{
  "condition": "category IN (?, ?, ?)",
  "parameters": ["science", "technology", "engineering"]
}
```

---

## Get Metadata by IDs

<span class="api-method post">POST</span> `/indices/{name}/metadata/get`

Get metadata for specific document IDs or by SQL condition.

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `string` | Index name |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `document_ids` | `array` | No | Specific document IDs |
| `condition` | `string` | No | SQL WHERE condition |
| `parameters` | `array` | No | Parameters for condition |
| `limit` | `int` | No | Maximum results |

Provide either `document_ids` or `condition`, not both.

=== "By IDs"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/metadata/get \
      -H "Content-Type: application/json" \
      -d '{"document_ids": [0, 1, 2]}'
    ```

=== "By Condition"

    ```bash
    curl -X POST http://localhost:8080/indices/my_index/metadata/get \
      -H "Content-Type: application/json" \
      -d '{
        "condition": "category = ?",
        "parameters": ["science"],
        "limit": 10
      }'
    ```

=== "Response"

    ```json
    {
      "metadata": [
        {"_id": 0, "title": "Document 1", "category": "science"},
        {"_id": 2, "title": "Document 3", "category": "science"}
      ],
      "count": 2
    }
    ```

---

## Metadata Schema

Metadata is stored as JSON in SQLite. Any valid JSON structure is supported:

```json
{
  "_id": 0,
  "title": "Document Title",
  "category": "science",
  "tags": ["ml", "nlp", "search"],
  "year": 2024,
  "score": 0.95,
  "published": true,
  "author": {
    "name": "John Doe",
    "email": "john@example.com"
  }
}
```

### Queryable Fields

Only top-level scalar fields can be queried:

- `title`, `category`, `year`, `score`, `published`

Nested objects and arrays are stored but not directly queryable.

### Best Practices

1. **Keep metadata flat** - Nested structures can't be filtered
2. **Use consistent types** - Same field should have same type across documents
3. **Index important fields** - Frequently filtered fields benefit from structure
4. **Limit size** - Large metadata increases memory usage
