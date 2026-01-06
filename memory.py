"""Memory module for conversation history with BM25S indexing."""

import json
import bm25s
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class ConversationMemory:
    """Manages conversation history with JSON persistence and BM25S indexing."""

    def __init__(
        self,
        history_file: str = "conversation_history.json",
        index_path: str = "bm25s_index_memory",
    ):
        self.history_file = Path(history_file)
        self.index_path = index_path
        self.history: List[Dict[str, str]] = []
        self.retriever: Optional[bm25s.BM25] = None
        self._load_history()

    def _load_history(self) -> None:
        """Load conversation history from JSON file."""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading history: {e}")
                self.history = []
        else:
            self.history = []

    def _save_history(self) -> None:
        """Save conversation history to JSON file."""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"Error saving history: {e}")

    def add_exchange(self, question: str, response: str) -> None:
        """
        Add a question-response exchange to the history.

        Args:
            question: The user's question
            response: The assistant's response
        """
        timestamp = datetime.now().isoformat()
        exchange = {
            "timestamp": timestamp,
            "question": question,
            "response": response,
        }
        self.history.append(exchange)
        self._save_history()
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the BM25S index from conversation history."""
        if not self.history:
            return

        # Create corpus from conversation history
        corpus = []
        for exchange in self.history:
            # Combine question and response for better context retrieval
            text = f"Q: {exchange['question']}\nA: {exchange['response']}"
            corpus.append(text)

        # Tokenize and index
        corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
        self.retriever = bm25s.BM25(corpus=corpus)
        self.retriever.index(corpus_tokens)

        # Save index to disk
        try:
            self.retriever.save(self.index_path)
        except Exception as e:
            print(f"Error saving index: {e}")

    def search(self, query: str, top_k: int = 2) -> str:
        """
        Search conversation history using BM25S keyword matching.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            Formatted string containing relevant conversation history
        """
        if not self.history:
            return ""

        # Ensure index is loaded
        if self.retriever is None:
            try:
                self.retriever = bm25s.BM25.load(self.index_path, load_corpus=True)
            except (FileNotFoundError, Exception):
                self._rebuild_index()

        if self.retriever is None:
            return ""

        # Search the index
        query_tokens = bm25s.tokenize(query, show_progress=False)
        try:
            docs, _ = self.retriever.retrieve(query_tokens, k=top_k)
        except ValueError:
            docs, _ = self.retriever.retrieve(query_tokens, k=1)

        # Format results
        if len(docs[0]) == 0:
            return ""

        context_parts = []
        for doc in docs[0]:
            if doc:
                context_parts.append(str(doc))

        return "\n---\n\n".join(context_parts)

    def get_recent_history(self, n: int = 5) -> str:
        """
        Get the most recent conversation exchanges.

        Args:
            n: Number of recent exchanges to retrieve

        Returns:
            Formatted string containing recent conversation history
        """
        if not self.history:
            return ""

        recent = self.history[-n:]
        context_parts = []

        for exchange in recent:
            text = f"Q: {exchange['question']}\nA: {exchange['response']}"
            context_parts.append(text)

        return "\n---\n\n".join(context_parts)

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self.history = []
        self._save_history()
        if Path(self.index_path).exists():
            import shutil

            shutil.rmtree(self.index_path)
