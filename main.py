import os
import dspy
import bm25s
from lexical import (
    tokenize_and_index_corpus,
    read_markdown_files,
    load_index,
    search_index,
)
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "bm25s_index"
MD_FILES = os.getenv("MD_FILES")

lm = dspy.LM('azure_ai/gpt-5-mini')
dspy.configure(lm=lm)


def get_retriever(refresh_index: bool = False) -> bm25s.BM25:
    if refresh_index:
        md_files = read_markdown_files(MD_FILES)
        tokenize_and_index_corpus(md_files, index_path=INDEX_PATH)
    try:
        retriever = load_index(INDEX_PATH)
    except FileNotFoundError:
        md_files = read_markdown_files(MD_FILES)
        tokenize_and_index_corpus(md_files, index_path=INDEX_PATH)
        retriever = load_index(INDEX_PATH)
    return retriever


def search(query: str, top_k: int = 2):
    retriever = get_retriever()
    docs, _ = search_index(retriever, query, top_k=top_k)
    doc_list = [str(doc) for doc in docs[0]]
    context = "\n---\n\n".join(doc_list)
    return context


class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question)
        return self.respond(context=context, question=question)


def main():
    rag = RAG()
    question = "What is emacs?"
    response = rag(question=question)
    print(response.response)


if __name__ == "__main__":
    main()
