import chromadb
from sentence_transformers import SentenceTransformer
from app.core.config import settings


class RAGService:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name="korean_reading",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    def index_content(self, content_id: str, passage: str, question: str, answer: str) -> int:
        chunks = self._chunk(passage)
        embeddings = self.embedder.encode(chunks).tolist()

        self.collection.upsert(
            ids=[f"{content_id}_{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"content_id": content_id} for _ in chunks],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self.collection.count()),
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

    def _chunk(self, text: str, size: int = 300, overlap: int = 50) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            chunks.append(text[start:start + size])
            start += size - overlap
        return chunks


rag_service = RAGService()
