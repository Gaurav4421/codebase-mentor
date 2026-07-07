"""
FAISS + embeddings. Unchanged behavior from the notebook's `build_vectorstore`
(MMR search, bge-small embeddings) -- split out so vector indexing can be
tested/persisted independently of BM25 and the reranker.
"""
import logging
from dataclasses import dataclass
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from ingestion.chunker import CodeChunk

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def _embed_text_for_chunk(chunk: CodeChunk) -> str:
    """Richer text than the raw snippet (file/class/name/imports) so semantically
    similar code is found even when the snippet itself doesn't contain the
    words the user asks about. The raw snippet is what's shown to the LLM
    later (restored via metadata["code"]), not this enriched version."""
    header_parts = [f"File: {chunk.path}"]
    if chunk.parent_class:
        header_parts.append(f"Class: {chunk.parent_class}")
    if chunk.name:
        header_parts.append(f"Name: {chunk.name}")
    if chunk.file_imports:
        header_parts.append("Imports: " + ", ".join(chunk.file_imports[:8]))
    return "\n".join(header_parts) + "\n\n" + chunk.content


def chunk_to_document(chunk: CodeChunk) -> Document:
    metadata = {
        "path": chunk.path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "type": chunk.chunk_type,
        "language": chunk.language,
        "name": chunk.name,
        "parent_class": chunk.parent_class,
        "code": chunk.content,  # raw snippet, restored after vector search
    }
    return Document(page_content=chunk.content, metadata=metadata)


@dataclass
class VectorIndex:
    store: FAISS
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL

    def as_retriever(self, k: int = 12, search_type: str = "mmr"):
        return self.store.as_retriever(search_type=search_type, search_kwargs={"k": k})

    def save(self, path: str) -> None:
        self.store.save_local(path)

    @classmethod
    def load(cls, path: str, embedding_model_name: str = DEFAULT_EMBEDDING_MODEL) -> "VectorIndex":
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        store = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
        return cls(store=store, embedding_model_name=embedding_model_name)


def build_vector_index(
    docs: List[Document],
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> VectorIndex:
    """`docs` should be the *enriched* embed-documents (see chunk_to_document +
    _embed_text_for_chunk usage in hybrid_retriever.build_indexes) -- kept as a
    plain function over Documents here so this module has no opinion about
    where those documents came from."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    store = FAISS.from_documents(docs, embeddings)
    logger.info("Built FAISS index over %d documents", len(docs))
    return VectorIndex(store=store, embedding_model_name=embedding_model_name)
