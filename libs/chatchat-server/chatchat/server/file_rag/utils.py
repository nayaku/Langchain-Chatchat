from chatchat.server.file_rag.retrievers import (
    BaseRetrieverService,
    EnsembleRetrieverService,
    VectorstoreRetrieverService,
    MilvusVectorstoreRetrieverService,
    VectorstoreRetrieverServiceMMR,
)

Retrivals = {
    "milvusvectorstore": MilvusVectorstoreRetrieverService,
    "vectorstore": VectorstoreRetrieverService,
    "ensemble": EnsembleRetrieverService,
    "vectorstoreMMR": VectorstoreRetrieverServiceMMR
}


def get_Retriever(type: str = "vectorstore") -> BaseRetrieverService:
    return Retrivals[type]
