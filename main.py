from fastapi import FastAPI
from routers.upload_router import upload_router
from routers.query_router import query_router
from routers.websocket_router import websocket_router

app = FastAPI(title="OOP-Based RAG Chatbot")

app.include_router(upload_router)
app.include_router(query_router)
app.include_router(websocket_router)

@app.get("/")
def root():
    return {"message": "Chatbot is live with OOP support."}
