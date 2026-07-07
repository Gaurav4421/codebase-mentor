"""
CrossEncoder reranking, unchanged from the notebook. The model is lazily
loaded and cached at module level (it's expensive to construct) -- wrapped in
a small class instead of a bare global so tests can construct a Reranker with
a fake/mocked model without touching module globals.
"""
import logging
from typing import List

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, model: CrossEncoder = None):
        self.model_name = model_name
        self._model = model 

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading reranker model %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: List[Document], top_k: int = 5) -> List[Document]:
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _score in ranked[:top_k]]
