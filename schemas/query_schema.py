from pydantic import BaseModel
from typing import List

class PDFQueryInput(BaseModel):
    question: str

class PDFQueryResult(BaseModel):
    query: str
    chunks_used: int
    answer: str
    snippets: List[str]

