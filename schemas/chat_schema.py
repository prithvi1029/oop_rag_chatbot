from pydantic import BaseModel
from typing import Literal

class ChatQuery(BaseModel):
    query: str

class ChatResponse(BaseModel):
    route: Literal["sap", "code", "web", "doc"]
    response: str
    timestamp: str
