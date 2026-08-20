# ColGREP stdio server

`colgrep serve --stdio [PATH] [--model MODEL]` loads one existing index and reads version-1 newline-delimited JSON from stdin. Responses are written to stdout; no network socket is opened. The process holds the index lock, so stop it before updating the index. If the index is missing, startup fails explicitly without creating one, allowing clients to run `colgrep init` before retrying.

Requests use a scalar string or number `id`:

```json
{"version":1,"id":"health","op":"health"}
{"version":1,"id":"search-1","op":"search","query":"database pooling","top_k":10}
{"version":1,"id":"stop","op":"shutdown"}
```

Search also accepts:

- `semantic_only` and `code_only` booleans;
- `include` and `exclude` glob arrays;
- `restrict_to_dir`, a relative directory contained by the served root.

Successful search responses contain the same serialized results as `colgrep --json`. Errors use:

```json
{"version":1,"id":"search-1","ok":false,"op":"search","error":{"code":"invalid_request","message":"..."}}
```

The protocol bounds requests to 64 KiB, responses to 8 MiB, queries to 4096 bytes, results to 100, and glob arrays to 16 patterns of 256 bytes each and 256 brace-expanded forms total. Restart the process after changing the index or model.
