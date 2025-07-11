"""
Phase 5: LLM Generation Service

This module implements the LLM generation phase using ChatGroq API integration
with LangChain, as specified in the development plan.

Key Components:
- ChatGroq LLM integration with Llama-3.1-8b-instant
- Prompt templates for system and document formatting  
- FastAPI service endpoint /ask
- Integration with Phase 4 retrieval system

Features:
- Temperature=0 for deterministic, fact-focused answers
- Timestamp citation in generated responses
- Source metadata tracking
- Integration with existing pipeline
"""

from .qa_service import QAService, QARequest, QAResponse
from .prompt_templates import SystemPrompt, DocumentPrompt, create_qa_chain
from .llm_integration import ChatGroqLLM

__all__ = [
    "QAService",
    "QARequest", 
    "QAResponse",
    "SystemPrompt",
    "DocumentPrompt", 
    "create_qa_chain",
    "ChatGroqLLM"
]

__version__ = "1.0.0" 