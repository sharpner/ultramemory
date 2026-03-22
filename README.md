# ultramemory

Portable knowledge graph builder for local documents. Single Go binary, no cloud, no setup.

## Dependencies

- **Go** (build time)
- **Ollama** (runtime) with `gemma3:4b` and `nomic-embed-text` pulled

```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text
```

## Install

```bash
go install github.com/sharpner/ultramemory@latest
```

Or build from source:

```bash
go build -o ultramemory .
```

## Usage

```bash
# Ingest a directory and start the worker
ultramemory run ./my-docs

# Or separately:
ultramemory ingest ./my-docs   # enqueue files into SQLite job queue
ultramemory worker             # process queue (runs until Ctrl+C)

# Search the graph
ultramemory search "Alice Schmidt TechCorp"

# Status
ultramemory status
```

## Environment

| Variable           | Default                   | Description                      |
|--------------------|---------------------------|----------------------------------|
| `MEMORY_DB`        | `memory-local.db`         | SQLite database path             |
| `MEMORY_OLLAMA`    | `http://localhost:11434`  | Ollama base URL                  |
| `MEMORY_MODEL`     | `gemma3:4b`               | Entity/edge extraction model     |
| `MEMORY_EMBED_MODEL` | `nomic-embed-text`      | Embedding model                  |
| `MEMORY_GROUP`     | `default`                 | Namespace for graph isolation    |

## How it works

1. **Ingest**: Files are chunked (1500 chars, 150 overlap) and enqueued as SQLite jobs
2. **Extract**: gemma3:4b extracts named entities and relationships from each chunk
3. **Embed**: nomic-embed-text generates 768-dim vectors for semantic search
4. **Search**: Hybrid FTS5 + cosine similarity with RRF fusion

Max 1 concurrent gemma3:4b call — resource-friendly on consumer hardware.

## Acknowledgements

Inspired by [Graphiti](https://github.com/getzep/graphiti) — a full-featured Python knowledge graph library by [Zep AI](https://github.com/getzep). ultramemory ports the core episode→entity→edge model to a single, dependency-free Go binary.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
