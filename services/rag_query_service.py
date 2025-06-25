from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from utils.mongo_helper import get_collection
import asyncio

class RAGQueryEngine:
    def __init__(self):
        self.index_path = "faiss_index/sample_index"
        self.collection = get_collection("rag_query_logs")
        self.llm = ChatGroq(
            model_name="Llama3-8b-8192",
            api_key="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",
            temperature=0
        )

    async def handle_query(self, question: str):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        faiss_index = FAISS.load_local(self.index_path, embeddings, allow_dangerous_deserialization=True)

        docs = await asyncio.to_thread(faiss_index.similarity_search, question, 3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Use the following document excerpts to answer the question:\n\n{context}\n\nUser Question: {question}"

        answer = await asyncio.to_thread(self.llm.invoke, prompt)

        self.collection.insert_one({
            "question": question,
            "index_path": self.index_path,
            "chunks_used": len(docs),
            "document_snippets": [d.page_content for d in docs],
            "answer": answer.content if hasattr(answer, "content") else str(answer),
            "timestamp": datetime.utcnow()
        })

        return {
            "query": question,
            "chunks_used": len(docs),
            "answer": answer.content if hasattr(answer, "content") else str(answer),
            "snippets": [d.page_content for d in docs]  # ✅ Added this line
        }
