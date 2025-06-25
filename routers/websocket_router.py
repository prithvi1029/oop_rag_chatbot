from fastapi import APIRouter, WebSocket
from services.websocket_service import WebSocketAgentHandler

websocket_router = APIRouter()

@websocket_router.websocket("/chat")
async def chat(websocket: WebSocket):
    handler = WebSocketAgentHandler(websocket)
    await handler.run()
