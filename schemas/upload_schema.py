from pydantic import BaseModel

class UploadResponse(BaseModel):
    message: str
    index_path: str
    chunks: int
