# Notes RAG

Local retrieval-augmented generation (RAG) playground that indexes Markdown notes with BM25 and answers questions through [DSPy](https://github.com/stanfordnlp/dspy). Markdown files are recursively scanned, tokenized, and stored in a disk-backed BM25 index so you can refresh or query the corpus without re-ingesting every time.

## Features
- Recursive Markdown loader with basic hygiene filters (skips `creds`, `untitled`, empty files)
- Disk-persisted BM25 index built via `bm25s`
- **Conversation memory with BM25S indexing**: All conversations are automatically saved to JSON and indexed for keyword-based retrieval
- Simple DSPy `ChainOfThought` responder with conversation history context
- Lightweight API you can import: refresh the index, run BM25 search, or compose your own RAG flow
- Interactive TUI (Terminal User Interface) built with Textual for real-time querying

## Project layout
```
lexical.py        # Markdown IO, BM25 indexing helpers
main.py           # RAG entry point using DSPy + Azure GPT-5-mini
memory.py         # Conversation history with JSON persistence and BM25S indexing
tui.py            # Interactive terminal UI for querying notes
styles.css        # CSS styling for the TUI
pyproject.toml    # Project metadata and dependencies
```

## Requirements
- Python 3.11+
- An Azure OpenAI (or Azure AI) GPT-5 deployment accessible via DSPy
- A directory of Markdown notes you want to index (default path lives in `main.py`)
- Textual library for the interactive TUI (included in dependencies)

## Setup
1. Clone the repo and switch into it.
2. Create a virtual environment (or rely on `uv` / `pip`):
	```bash
	uv sync  # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
	```
	The `pyproject.toml` already declares runtime deps (`bm25s`, `dspy`, `pystemmer`, `python-dotenv`).
3. Create a `.env` file in the project root with the credentials your DSPy configuration expects, for example:
	```bash
	AZURE_OPENAI_ENDPOINT=https://<your-endpoint>.com/
	AZURE_OPENAI_API_KEY=<api-key>
	AZURE_OPENAI_DEPLOYMENT=gpt-5-mini
 	MD_FILES=<markdown-file-path>
	```
	Adjust the variable names to match your DSPy / Azure setup. `dotenv` loads this file before the model is configured.

## Preparing the knowledge base
- Point `MD_FILES` (see `main.py`) at the root directory that stores your Markdown notes.
- `lexical.read_markdown_files()` walks every subdirectory, skips unwanted filenames, and returns a `{relative_path: content}` map.
- The BM25 index is stored under `INDEX_PATH` (default `bm25s_index`). Delete that folder or pass `refresh_index=True` to rebuild from scratch.

## Running the RAG demo

### Command-line demo
```bash
uv run python main.py
```
Steps performed:
1. Load `.env` and configure DSPy to use the Azure GPT-5-mini deployment.
2. Load the on-disk BM25 index; if it does not exist, Markdown files are read and indexed first.
3. Instantiate the `RAG` module (BM25 retrieval + `ChainOfThought` responder).
4. Ask the hard-coded sample query (`"What are my TODOs?"`) and print the model's response.

### Interactive TUI
For a more user-friendly experience, run the interactive terminal UI:
```bash
uv run python tui.py
```
This launches a Textual-based interface where you can:
- Type questions in the input field
- See responses displayed in real-time
- Navigate with keyboard shortcuts

**Note**: The TUI includes a workaround for multiprocessing compatibility issues with `tqdm` (used by `bm25s`) in async/threading contexts. If you encounter `ValueError: bad value(s) in fds_to_keep` errors, ensure the `multiprocessing.synchronize` module is mocked before imports as shown in `tui.py`.

## Conversation memory
Every conversation is automatically persisted and indexed for future context retrieval:

- **JSON persistence**: All Q&A exchanges are saved to `conversation_history.json` with timestamps
- **BM25S indexing**: Conversation history is tokenized and indexed using the same BM25S approach as the knowledge base
- **Keyword-based search**: Past conversations are retrieved using keyword matching (not semantic search)
- **Automatic context**: Relevant conversation history is automatically included in the LLM prompt alongside knowledge base results

### How it works
1. User asks a question
2. System searches both the knowledge base AND conversation history using BM25S
3. LLM receives combined context (notes + relevant past conversations)
4. Response is generated and automatically saved to memory
5. BM25S index is updated for future queries

### Memory management
```python
from memory import ConversationMemory

# Access the conversation memory
memory = ConversationMemory()

# Search past conversations
results = memory.search("Python", top_k=3)

# Get recent history
recent = memory.get_recent_history(n=5)

# Clear all history
memory.clear_history()
```

See [MEMORY_README.md](MEMORY_README.md) for detailed documentation on the conversation memory system.

## Programmatic usage
- Refresh the index manually: call `get_retriever(refresh_index=True)`.
- Retrieve documents only: `docs, scores = search_index(retriever, query, top_k=5)`.
- Plug the retriever into your own DSPy modules: import `search` or `RAG` from `main.py`, or instantiate `dspy.ChainOfThought` with any context you build.

## Next steps
- Add evaluation harnesses (e.g., DSPy metrics) or guardrails around responses.
- Enhance the TUI with streaming responses or advanced result filtering.
- Implement different retrieval strategies or hybrid search approaches.
- Add conversation export/import functionality.
