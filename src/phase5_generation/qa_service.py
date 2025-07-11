"""
Phase 5: QA Service with FastAPI

Implements the FastAPI endpoint /ask as specified in the development plan:
- JSON response with {"answer": str, "sources": List[metadata]}
- Integration with ChatGroq LLM and Phase 4 retrieval
- Async/streaming support if applicable
- Reasonable latency response with LLM answer + source list

Key Features:
- FastAPI /ask endpoint
- Integration with existing Phase 4 retrieval
- Proper error handling and validation
- Source metadata tracking
- Async processing for better performance
"""

import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Add Phase 4 to path for retrieval access  
sys.path.insert(0, str(Path(__file__).parent.parent / "phase4_retriever"))

from .llm_integration import ChatGroqLLM
from .prompt_templates import format_documents_for_prompt

# Import Phase 4 retrieval
try:
    from retriever import Retriever
    from models import Document
    PHASE4_AVAILABLE = True
except ImportError:
    try:
        from src.phase4_retriever.retriever import Retriever
        from src.phase4_retriever.models import Document
        PHASE4_AVAILABLE = True
    except ImportError:
        print("⚠️  Phase 4 retrieval not available")
        Retriever = None
        Document = None
        PHASE4_AVAILABLE = False

logger = logging.getLogger(__name__)


# Request/Response Models
class QARequest(BaseModel):
    """Request model for QA endpoint."""
    
    question: str = Field(..., description="Natural language question about video content")
    k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve for context")
    video_id: Optional[str] = Field(None, description="Optional filter by specific video ID")
    include_visual: bool = Field(True, description="Include visual frame segments in retrieval")
    include_audio: bool = Field(True, description="Include audio transcript segments in retrieval")
    
    @validator('question')
    def question_must_not_be_empty(cls, v):
        """Ensure question is not empty."""
        if not v or not v.strip():
            raise ValueError('question cannot be empty')
        return v.strip()


class SourceMetadata(BaseModel):
    """Source metadata for citations."""
    
    video_id: str = Field(..., description="Video identifier")
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds") 
    modality: str = Field(..., description="Content type: audio or frame")
    content_preview: Optional[str] = Field(None, description="Preview of content")


class QAResponse(BaseModel):
    """Response model for QA endpoint as specified in development plan."""
    
    answer: str = Field(..., description="Generated answer with timestamp citations")
    sources: List[SourceMetadata] = Field(..., description="Source metadata for all retrieved segments")
    
    # Additional metadata
    metadata: Dict[str, Any] = Field(..., description="Generation metadata")
    processing_time_seconds: float = Field(..., description="Total processing time")
    retrieval_time_seconds: float = Field(..., description="Retrieval time")
    generation_time_seconds: float = Field(..., description="LLM generation time")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field("healthy", description="Service status")
    version: str = Field("1.0.0", description="Service version")
    components: Dict[str, bool] = Field(..., description="Component health status")
    uptime_seconds: float = Field(..., description="Service uptime")


class QAService:
    """
    Phase 5 QA Service Implementation
    
    Provides FastAPI service with /ask endpoint that integrates:
    - Phase 4 retrieval system
    - ChatGroq LLM generation  
    - Proper response formatting with sources
    """
    
    def __init__(self, 
                 retriever: Optional[Retriever] = None,
                 llm: Optional[ChatGroqLLM] = None,
                 persist_directory: str = "data/chroma"):
        """
        Initialize QA Service.
        
        Args:
            retriever: Optional Phase 4 Retriever instance
            llm: Optional ChatGroq LLM instance
            persist_directory: ChromaDB directory for retriever
        """
        
        self.start_time = time.time()
        
        # Initialize retriever
        if retriever is not None:
            self.retriever = retriever
        elif PHASE4_AVAILABLE:
            self.retriever = Retriever(persist_directory=persist_directory)
        else:
            self.retriever = None
            logger.warning("Phase 4 retriever not available")
        
        # Initialize LLM
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = ChatGroqLLM()
                # LLM initialized successfully - no logging needed
            except Exception as e:
                self.llm = None
                logger.error(f"Failed to initialize ChatGroq LLM: {e}")
        
        # QA Service initialized successfully
    
    async def process_question(self, request: QARequest) -> QAResponse:
        """
        Process question through retrieval + generation pipeline.
        
        Args:
            request: QA request with question and parameters
            
        Returns:
            QA response with answer and sources
            
        Raises:
            HTTPException: If processing fails
        """
        
        if not self.retriever:
            raise HTTPException(status_code=503, detail="Retrieval service unavailable")
        
        if not self.llm:
            raise HTTPException(status_code=503, detail="LLM service unavailable")
        
        start_time = time.time()
        
        try:
            # Phase 1: Retrieval
            retrieval_start = time.time()
            
            # Retrieve relevant documents
            documents = self.retriever.search(request.question, k=request.k)
            
            # Apply modality filters if specified
            if not request.include_audio:
                documents = [doc for doc in documents if not doc.is_audio_segment()]
            if not request.include_visual:
                documents = [doc for doc in documents if not doc.is_frame_segment()]
            
            # Apply video ID filter if specified
            if request.video_id:
                documents = [doc for doc in documents if doc.metadata.get('video_id') == request.video_id]
            
            retrieval_time = time.time() - retrieval_start
            
            if not documents:
                return QAResponse(
                    answer="I couldn't find any relevant video segments to answer your question.",
                    sources=[],
                    metadata={
                        "model": "N/A",
                        "temperature": 0,
                        "num_sources": 0,
                        "question": request.question
                    },
                    processing_time_seconds=time.time() - start_time,
                    retrieval_time_seconds=retrieval_time,
                    generation_time_seconds=0.0
                )
            
            # Phase 2: LLM Generation
            generation_start = time.time()
            
            llm_result = await self.llm.generate_response_async(
                question=request.question,
                context_documents=documents
            )
            
            generation_time = time.time() - generation_start
            
            # Build response
            sources = [
                SourceMetadata(
                    video_id=source["video_id"],
                    start=source["start"],
                    end=source["end"],
                    modality=source["modality"],
                    content_preview=source["content_preview"]
                )
                for source in llm_result["sources"]
            ]
            
            response = QAResponse(
                answer=llm_result["answer"],
                sources=sources,
                metadata=llm_result["metadata"],
                processing_time_seconds=time.time() - start_time,
                retrieval_time_seconds=retrieval_time,
                generation_time_seconds=generation_time
            )
            
            print(f"✅ Question processed in {response.processing_time_seconds:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process question: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    def get_health_status(self) -> HealthResponse:
        """Get service health status."""
        
        components = {
            "retriever": self.retriever is not None,
            "llm": self.llm is not None,
            "phase4_available": PHASE4_AVAILABLE
        }
        
        # Test LLM if available
        if self.llm:
            try:
                components["llm_responsive"] = self.llm.health_check()
            except:
                components["llm_responsive"] = False
        
        return HealthResponse(
            status="healthy" if all(components.values()) else "degraded",
            components=components,
            uptime_seconds=time.time() - self.start_time
        )


# Create FastAPI app
app = FastAPI(
    title="Video RAG QA Service",
    description="Phase 5: LLM Generation Service for Video RAG Pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global QA service instance
qa_service: Optional[QAService] = None


@app.on_event("startup")
async def startup_event():
    """Initialize QA service on startup."""
    global qa_service
    try:
        qa_service = QAService()
        logger.info("QA Service started successfully")
    except Exception as e:
        logger.error(f"Failed to start QA Service: {e}")
        qa_service = None


def get_qa_service() -> QAService:
    """Dependency to get QA service instance."""
    if qa_service is None:
        raise HTTPException(status_code=503, detail="QA Service not available")
    return qa_service


@app.post("/ask", response_model=QAResponse)
async def ask_question(
    request: QARequest,
    service: QAService = Depends(get_qa_service)
) -> QAResponse:
    """
    Main QA endpoint as specified in development plan.
    
    Processes natural language questions about video content and returns
    answers with timestamp citations and source metadata.
    """
    return await service.process_question(request)


@app.get("/health", response_model=HealthResponse) 
async def health_check(
    service: QAService = Depends(get_qa_service)
) -> HealthResponse:
    """Health check endpoint."""
    return service.get_health_status()


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Video RAG QA Service",
        "phase": "Phase 5 - LLM Generation",
        "version": "1.0.0",
        "endpoints": {
            "ask": "POST /ask - Process questions about video content",
            "health": "GET /health - Service health status"
        }
    }


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Server host
        port: Server port
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "src.phase5_generation.qa_service:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    # For direct execution
    run_server(reload=True) 