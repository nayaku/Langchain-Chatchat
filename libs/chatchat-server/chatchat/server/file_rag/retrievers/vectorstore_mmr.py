from __future__ import annotations

from langchain.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from chatchat.server.file_rag.retrievers.base import BaseRetrieverService


class VectorstoreRetrieverServiceMMR(BaseRetrieverService):
    def do_init(
            self,
            retriever: BaseRetriever = None,
            top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever

    @staticmethod
    def from_vectorstore(
            vectorstore: VectorStore,
            top_k: int,
            score_threshold: int | float,
    ):
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"lambda_mult": 0.25, "fetch_k": top_k * 5, "k": top_k},
        )
        return VectorstoreRetrieverServiceMMR(retriever=retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):
        return self.retriever.get_relevant_documents(query)[: self.top_k]
