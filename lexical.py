import bm25s
from pathlib import Path
from typing import Dict


def read_markdown_files(directory: str) -> Dict[str, str]:
    """
    Read all markdown files from a directory and return their contents.

    Args:
        directory (str): Path to the directory containing markdown files

    Returns:
        Dict[str, str]: Dictionary with filename as key and content as value
    """
    markdown_files = {}
    directory_path = Path(directory)

    if not directory_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist")

    if not directory_path.is_dir():
        raise NotADirectoryError(f"'{directory}' is not a directory")

    patterns = ("*.md", "*.markdown")

    # Walk subdirectories so deeply nested notes are also ingested
    for pattern in patterns:
        for file_path in directory_path.rglob(pattern):
            if file_path.is_dir():
                continue
            if "creds" in file_path.name.lower():
                continue  # Skip files containing 'creds' in their name
            if "untitled" in file_path.name.lower():
                continue  # Skip files containing 'untitled' in their name

            relative_name = file_path.relative_to(directory_path).as_posix()
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    if content == "":
                        continue  # Skip empty files
                    markdown_files[relative_name] = content
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return markdown_files


def tokenize_and_index_corpus(
    corpus_dict: Dict[str, str], index_path: str = "bm25s_index"
) -> None:
    corpus = list(corpus_dict.values())
    corpus_tokens = bm25s.tokenize(corpus, show_progress=False)
    retriever = bm25s.BM25(corpus=corpus)
    retriever.index(corpus_tokens)
    retriever.save(index_path)


def load_index(index_path: str) -> bm25s.BM25:
    retriever = bm25s.BM25.load(index_path, load_corpus=True)
    return retriever


def search_index(retriever: bm25s.BM25, query: str, top_k: int = 2) -> tuple:
    query_tokens = bm25s.tokenize(query, show_progress=False)
    docs, scores = retriever.retrieve(query_tokens, k=top_k)
    return docs, scores
