## Tavily Iterative Web-Retrieval RAG Demo

This project implements a small, purely-Python iterative web-retrieval RAG-style pipeline using the Tavily search API.

### Features

- **Iterative search loop** with a configurable maximum number of iterations (default: 3)
- **Tavily web search** to fetch fresh results
- **HTML fetching and text extraction** for each result URL
- **Passage chunking and TF-IDF ranking** for relevance
- **Heuristic context evaluation** to decide if more search is needed
- **Caching** so the same URL is not fetched twice in a run
- **Passage deduplication** using text hashing
- **Simple token/length budgeting** for the final context
- **Placeholder answering** that summarizes top passages and returns citations (URLs)

### Project Layout

The project follows a modern `src` layout for better modularity and package management.

```text
tavily_agent/
├── src/
│   ├── __init__.py
│   ├── cli.py               # CLI entry point
│   ├── pipeline.py          # Orchestration loop
│   ├── models.py            # Shared data classes
│   ├── agents/
│   │   ├── answerer.py      # LLM-based answering logic
│   │   └── evaluator.py     # Context sufficiency evaluation
│   ├── services/
│   │   └── web_search.py    # Tavily API & web fetching
│   └── processing/
│       └── text.py          # Chunking & ranking
├── tests/                       # Unit and integration tests
├── main.py                      # Root convenience script
├── pyproject.toml               # Package configuration
├── tavily_rag_agent.egg-info/   # Metadata
└── README.md
```

### Setup with uv

From the project root:

```bash
# create a virtual environment and install dependencies
uv venv
uv sync
```

Export your Tavily API key:

```bash
export TAVILY_API_KEY="your-api-key-here"
```

### Running the CLI

You can run the pipeline directly using `uv`:

```bash
uv run python main.py --question "What is Tavily and how does it work?"
```

Or use the installed entry point:

```bash
uv run tavily-rag --question "Latest overview of retrieval-augmented generation in 2025"
```

If `--question` is omitted, the program will prompt you for one on stdin.
