import os
import dspy
import bm25s
from lexical import (
    tokenize_and_index_corpus,
    read_markdown_files,
    load_index,
    search_index,
)
from memory import ConversationMemory
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = "bm25s_index"
MD_FILES = os.getenv("MD_FILES")

lm = dspy.LM("azure_ai/gpt-5-mini")
dspy.configure(lm=lm)

# Initialize conversation memory
conversation_memory = ConversationMemory()


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


def search_memory(query: str, top_k: int = 3) -> str:
    """Search conversation history using BM25S."""
    return conversation_memory.search(query, top_k=top_k)


class RagSignature(dspy.Signature):
    """You are a helpful AI assistant that uses provided knowledge 
    base and memory context to answer user questions.

    If a question cannot be answered using the provided context,
    let the user know that you don't have enough information to answer.
    
    """

    kb_context: str = dspy.InputField(description="Knowledge base context")
    memory_context: str = dspy.InputField(description="Conversation memory context")
    question: str = dspy.InputField(description="User's question")
    response: str = dspy.OutputField(description="Generated response to the user's")


class RAG(dspy.Module):
    def __init__(self):
        self.rag = dspy.ChainOfThought(signature=RagSignature)

    def forward(self, question, reset_memory: bool = False):
        if reset_memory:
            conversation_memory.clear_history()

        kb_context = search(question)
        memory_context = search_memory(question)

        response = self.rag(
            kb_context=kb_context, memory_context=memory_context, question=question
        )

        # Save to conversation history
        conversation_memory.add_exchange(question, response.response)

        return response


def main():
    rag = RAG()
    question = "What is emacs?"
    response = rag(question=question)
    print(response.response)


if __name__ == "__main__":
    main()
