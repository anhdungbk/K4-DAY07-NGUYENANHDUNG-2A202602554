from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin liên quan trong cơ sở tri thức."

        context = "\n\n".join(
            f"[{index}] Nguồn: {result['metadata'].get('source_url', result['metadata'].get('source', result['id']))}\n"
            f"{result['content']}"
            for index, result in enumerate(results, start=1)
        )
        prompt = (
            "Chỉ trả lời dựa trên ngữ cảnh bên dưới. Nếu ngữ cảnh không đủ, hãy nói rõ không tìm thấy thông tin. "
            "Khi dùng thông tin, hãy trích dẫn số nguồn trong ngoặc vuông.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            "Trả lời:"
        )
        return self.llm_fn(prompt)
