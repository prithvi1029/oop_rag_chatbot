import os, shutil
from datetime import datetime
from fastapi import UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS


from utils.mongo_helper import get_collection

class RAGUploader:
    def __init__(self):
        self.upload_dir = "uploaded_pdfs"
        self.index_dir = "faiss_index"
        self.collection = get_collection("pdf_upload_logs")

    async def handle_upload(self, file: UploadFile):
        os.makedirs(self.upload_dir, exist_ok=True)
        pdf_path = os.path.join(self.upload_dir, file.filename)
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            raise ValueError("No content found in PDF")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        index = FAISS.from_documents(chunks, embeddings)

        os.makedirs(self.index_dir, exist_ok=True)
        index_path = os.path.join(self.index_dir, file.filename.replace(".pdf", "_index"))
        index.save_local(index_path)

        self.collection.insert_one({
            "filename": file.filename,
            "pages": len(documents),
            "chunks": len(chunks),
            "index_path": index_path,
            "timestamp": datetime.utcnow()
        })

        return {"message": "Uploaded and indexed", "index_path": index_path, "chunks": len(chunks)}
