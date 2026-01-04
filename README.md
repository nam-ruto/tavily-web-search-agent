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

- `main.py` – CLI entry point and orchestration loop
- `web_retriever.py` – Tavily web search + page fetching and extraction
- `context_builder.py` – document chunking, ranking, and selection
- `evaluator.py` – heuristic context sufficiency evaluator + query refinement
- `answerer.py` – builds a context pack and returns an answer with citations
- `types.py` – shared data classes used across modules

### Setup with uv

From the project root:

```bash
cd "/Users/namhoang/Documents/2. Coding-space/ISODS/tavily_agent"

# (optional) ensure uv is installed
uv --version

# create a virtual environment and install dependencies
uv venv
uv sync
```

Export your Tavily API key:

```bash
export TAVILY_API_KEY="your-api-key-here"
```

### Running the CLI

You can run the pipeline directly with uv:

```bash
uv run python main.py --question "What is Tavily and how does it work?"
```

or rely on the installed script name (after `uv sync`):

```bash
uv run tavily-rag --question "Latest overview of retrieval-augmented generation in 2025"
```

If `--question` is omitted, the program will prompt you for one on stdin.


