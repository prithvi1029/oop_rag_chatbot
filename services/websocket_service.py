from fastapi import WebSocket
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from utils.mongo_helper import get_collection
from datetime import datetime
import asyncio, os, re, json
from typing import TypedDict


# ✅ Required TypedDict for LangGraph
class ChatState(TypedDict):
    query: str
    route: str
    response: str

class WebSocketAgentHandler:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.llm = ChatGroq(
            model_name="Llama3-8b-8192",
            api_key="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",
            temperature=0
        )
        os.environ["TAVILY_API_KEY"] = "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"
        self.web_search = TavilySearch(api_key=os.getenv("TAVILY_API_KEY"), max_results=3)
        self.collection = get_collection("chat_logs")
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ChatState)
        graph.add_node("router", self.route_node)
        graph.add_node("sap", self.sap_node)
        graph.add_node("code", self.code_node)
        graph.add_node("doc", self.doc_node)
        graph.add_node("web", self.web_node)

        graph.set_entry_point("router")
        graph.add_conditional_edges("router", lambda state: state["route"], {
            "sap": "sap", "code": "code", "doc": "doc", "web": "web"
        })
        for node in ["sap", "code", "doc", "web"]:
            graph.add_edge(node, END)

        return graph.compile()

    async def route_node(self, state: ChatState) -> ChatState:
        prompt = (
            "You are a routing agent. Classify the user's query into one of the following: "
            "'sap', 'code', 'web', or 'doc'. Respond with only one word.\n\n"
            f"Query: {state['query']}"
        )
        result = await asyncio.to_thread(self.llm.invoke, prompt)
        decision = result.content.strip().lower()
        match = re.search(r"\b(sap|code|web|doc)\b", decision)
        return {
            "query": state["query"],
            "route": match.group(1) if match else "web",
            "response": ""
        }

    async def sap_node(self, state: ChatState) -> ChatState:
        result = await asyncio.to_thread(self.llm.invoke, state["query"])
        return {**state, "response": result.content}

    async def code_node(self, state: ChatState) -> ChatState:
        prompt = f"You are a code reviewer. Improve the following code:\n{state['query']}"
        result = await asyncio.to_thread(self.llm.invoke, prompt)
        return {**state, "response": result.content}

    async def doc_node(self, state: ChatState) -> ChatState:
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        faiss_index = FAISS.load_local("faiss_index/sample_index", embedding, allow_dangerous_deserialization=True)
        docs = await asyncio.to_thread(faiss_index.similarity_search, state["query"], 3)
        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"Answer using this document:\n{context}\n\nQuery: {state['query']}"
        result = await asyncio.to_thread(self.llm.invoke, prompt)
        return {**state, "response": result.content}

    async def web_node(self, state: ChatState) -> ChatState:
        results = await asyncio.to_thread(self.web_search.invoke, state["query"])
        formatted = "\n\n".join([f"{r['title']}\n{r['content']}" for r in results.get("results", [])])
        prompt = f"Answer using this info:\n{formatted}\n\nQuery: {state['query']}"
        result = await asyncio.to_thread(self.llm.invoke, prompt)
        return {**state, "response": result.content}

    async def run(self):
        await self.websocket.accept()
        try:
            while True:
                raw = await self.websocket.receive_text()

                # ✅ Accept JSON or plain text
                try:
                    data = json.loads(raw)
                    query = data.get("query", "").strip()
                except json.JSONDecodeError:
                    query = raw.strip()

                if not query:
                    await self.websocket.send_text("⚠️ Error: Missing 'query'")
                    continue

                print(f"🟡 Received: {query}")
                state: ChatState = {
                    "query": query,
                    "route": "",
                    "response": ""
                }

                try:
                    final = await self.graph.ainvoke(state)
                    print(f"🧭 Routed to: {final['route']}")
                    self.collection.insert_one({
                        "query": final["query"],
                        "route": final["route"],
                        "response": final["response"],
                        "timestamp": datetime.utcnow()
                    })
                    await self.websocket.send_text(final["response"])
                except Exception as e:
                    await self.websocket.send_text(f"⚠️ Internal error: {str(e)}")

        except Exception as e:
            print(f"❌ WebSocket error: {e}")
        finally:
            await self.websocket.close()
