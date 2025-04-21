from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    GPT4_1_NANO = "gpt-4.1-nano"
    GPT4_O_MINI = "gpt-4o-mini"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GPT4_1_NANO)

class SourceName(str, Enum):
    RAG = "rag"
    GOOGLE = "google"
class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName
    source: SourceName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int