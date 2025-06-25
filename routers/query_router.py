from fastapi import APIRouter, Query
from services.rag_query_service import RAGQueryEngine
from schemas.query_schema import PDFQueryResult

query_router = APIRouter()

@query_router.get("/query-pdf", response_model=PDFQueryResult)
async def query_pdf(question: str = Query(...)):
    service = RAGQueryEngine()
    return await service.handle_query(question)
