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

## JSON API

Both `search` and `status` support `-format json` for machine-readable output.

### search -format json

Each result is emitted as a newline-delimited JSON object (one per line):

```bash
ultramemory search -format json "Alice Schmidt TechCorp"
```

```json
{"rank":1,"type":"entity","title":"Alice Schmidt","body":"person","score":0.0312}
{"rank":2,"type":"edge","title":"WORKS_AT","body":"Alice Schmidt works at TechCorp Berlin","score":0.0294}
{"rank":3,"type":"entity","title":"TechCorp Berlin","body":"organisation","score":0.0278}
```

| Field   | Type    | Description                                      |
|---------|---------|--------------------------------------------------|
| `rank`  | int     | 1-based result rank                              |
| `type`  | string  | `"entity"` or `"edge"`                           |
| `title` | string  | Entity name or edge relation type                |
| `body`  | string  | Entity type (for entities) or fact (for edges)   |
| `score` | float64 | RRF fusion score (higher = more relevant)        |

### status -format json

```bash
ultramemory status -format json
```

```json
{
  "graph": {"episodes": 42, "entities": 187, "edges": 312},
  "queue": {"pending": 0, "processing": 0, "done": 156, "failed": 2}
}
```

| Field                | Type | Description                              |
|----------------------|------|------------------------------------------|
| `graph.episodes`     | int  | Number of ingested text chunks           |
| `graph.entities`     | int  | Named entities extracted                 |
| `graph.edges`        | int  | Relationships extracted                  |
| `queue.pending`      | int  | Chunks waiting for extraction            |
| `queue.processing`   | int  | Chunks currently being processed         |
| `queue.done`         | int  | Successfully processed chunks            |
| `queue.failed`       | int  | Chunks that failed extraction            |

## How it works

1. **Ingest**: Files are chunked (1500 chars, 150 overlap) and enqueued as SQLite jobs
2. **Extract**: gemma3:4b extracts named entities and relationships from each chunk
3. **Embed**: nomic-embed-text generates 768-dim vectors for semantic search
4. **Search**: Hybrid FTS5 + cosine similarity fused via RRF, then extended by MAGMA graph traversal

### Search pipeline detail

```
Query
 ├─ FTS5 (full-text search on entities + edges)
 ├─ Vector search (cosine similarity, threshold 0.3)
 ├─ RRF fusion (Reciprocal Rank Fusion, k=60)
 └─ MAGMA graph traversal (arxiv.org/abs/2601.03236)
      ├─ Top-5 RRF entities become seed nodes
      ├─ Beam search (width=10, depth=5, budget=200 nodes)
      ├─ Transition: S = exp(λ₁·φ(edge_type, intent) + λ₂·cos_sim)
      │    ├─ Intent classified from query keywords (Why/When/Entity/What)
      │    └─ Edge types routed: Causal/Temporal/Relational/Attributive
      └─ Graph-traversal results appended after direct RRF matches
```

Max 1 concurrent gemma3:4b call — resource-friendly on consumer hardware.

## Acknowledgements

Inspired by [Graphiti](https://github.com/getzep/graphiti) — a full-featured Python knowledge graph library by [Zep AI](https://github.com/getzep). ultramemory ports the core episode→entity→edge model to a single, dependency-free Go binary.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
