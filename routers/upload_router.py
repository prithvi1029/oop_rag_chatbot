from fastapi import APIRouter, UploadFile, File, HTTPException
from services.rag_upload_service import RAGUploader
from schemas.upload_schema import UploadResponse

upload_router = APIRouter()

@upload_router.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    try:
        service = RAGUploader()
        return await service.handle_upload(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
