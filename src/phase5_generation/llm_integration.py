"""
Phase 5: ChatGroq LLM Integration

Implements ChatGroq LLM integration as specified in the development plan:
- Temperature=0 for deterministic answers
- Proper error handling and retries
- Integration with Phase 4 Document format
- Source extraction and citation formatting

Uses ChatGroq with Llama models for cost-effective, high-quality generation.
"""

import os
import time
import tiktoken
import logging
from typing import List, Optional, Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ChatGroqLLM:
    """
    ChatGroq LLM integration for Phase 5 generation.
    
    Handles ChatGroq API configuration, initialization, and interaction
    as specified in the development plan.
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "llama-3.1-8b-instant",
                 temperature: float = 0.0,
                 max_tokens: Optional[int] = None,
                 max_retries: int = 3,
                 timeout: float = 30.0):
        """
        Initialize ChatGroq LLM.
        
        Args:
            api_key: GROQ API key (defaults to GROQ_API_KEY env var)
            model: Model name (default: llama-3.1-8b-instant)
            temperature: Temperature for generation (default: 0 for deterministic)
            max_tokens: Maximum tokens to generate
            max_retries: Number of retries on API errors
            timeout: Request timeout in seconds
        """
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in environment variables or pass as parameter."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Initialize ChatGroq instance
        try:
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                timeout=timeout
            )
            # Initialization successful - no logging needed for clean output
            
        except Exception as e:
            logger.error(f"Failed to initialize ChatGroq LLM: {e}")
            raise RuntimeError(f"ChatGroq initialization failed: {e}")
    
    def generate_response(self, 
                         question: str, 
                         context_documents: List,
                         system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response using ChatGroq LLM with retrieved documents.
        
        Args:
            question: User question
            context_documents: List of Document objects from Phase 4 retrieval
            system_prompt: Optional custom system prompt
            
        Returns:
            Dict with 'answer', 'sources', and metadata
        """
        
        try:
            # Import here to avoid circular dependency
            from .prompt_templates import format_documents_for_prompt, SYSTEM_PROMPT_TEMPLATE
            
            # Format documents for prompt context
            formatted_context = format_documents_for_prompt(context_documents)
            
            # Use provided system prompt or default
            prompt_template = system_prompt or SYSTEM_PROMPT_TEMPLATE
            
            # Create the full prompt
            full_prompt = prompt_template.format(
                question=question,
                context=formatted_context
            )
            
            # Enhanced debugging: Show comprehensive prompt details
            _debug_prompt_details(full_prompt, context_documents, question)
            
            # Generate response
            print(f"🔄 Generating response with ChatGroq...")
            response = self.llm.invoke([HumanMessage(content=full_prompt)])
            
            # Extract sources from context documents
            sources = []
            for doc in context_documents:
                source_info = {
                    "video_id": doc.metadata.get("video_id"),
                    "start": doc.metadata.get("start"),
                    "end": doc.metadata.get("end"),
                    "modality": doc.metadata.get("modality"),
                    "content_preview": doc.page_content[:100] if doc.page_content else None
                }
                sources.append(source_info)
            
            result = {
                "answer": response.content,
                "sources": sources,
                "metadata": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "num_sources": len(context_documents),
                    "question": question
                }
            }
            
            print(f"✅ Generated response ({len(response.content)} characters)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")
    
    def generate_simple_response(self, prompt: str) -> str:
        """
        Generate simple response for direct prompt.
        
        Args:
            prompt: Direct prompt text
            
        Returns:
            Generated response text
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Failed to generate simple response: {e}")
            raise RuntimeError(f"Simple LLM generation failed: {e}")
    
    async def generate_response_async(self, 
                                    question: str, 
                                    context_documents: List,
                                    system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Async version of generate_response for FastAPI integration.
        
        Args:
            question: User question
            context_documents: List of Document objects from Phase 4 retrieval
            system_prompt: Optional custom system prompt
            
        Returns:
            Dict with 'answer', 'sources', and metadata
        """
        
        try:
            # Import here to avoid circular dependency
            from .prompt_templates import format_documents_for_prompt, SYSTEM_PROMPT_TEMPLATE
            
            # Format documents for prompt context
            formatted_context = format_documents_for_prompt(context_documents)
            
            # Use provided system prompt or default
            prompt_template = system_prompt or SYSTEM_PROMPT_TEMPLATE
            
            # Create the full prompt
            full_prompt = prompt_template.format(
                question=question,
                context=formatted_context
            )
            
            # Enhanced debugging: Show comprehensive prompt details
            _debug_prompt_details(full_prompt, context_documents, question)
            
            # Generate response using async invoke
            print(f"🔄 Generating response with ChatGroq...")
            response = await self.llm.ainvoke([HumanMessage(content=full_prompt)])
            
            # Extract sources from context documents
            sources = []
            for doc in context_documents:
                source_info = {
                    "video_id": doc.metadata.get("video_id"),
                    "start": doc.metadata.get("start"),
                    "end": doc.metadata.get("end"),
                    "modality": doc.metadata.get("modality"),
                    "content_preview": doc.page_content[:100] if doc.page_content else None
                }
                sources.append(source_info)
            
            result = {
                "answer": response.content,
                "sources": sources,
                "metadata": {
                    "model": self.model,
                    "temperature": self.temperature,
                    "num_sources": len(context_documents),
                    "question": question
                }
            }
            
            print(f"✅ Generated response ({len(response.content)} characters)")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate async response: {e}")
            raise RuntimeError(f"Async LLM generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "api_configured": bool(self.api_key)
        }
    
    def health_check(self) -> bool:
        """
        Perform health check by making a simple API call.
        
        Returns:
            True if LLM is healthy and responsive
        """
        
        try:
            test_response = self.llm.invoke([HumanMessage(content="Hello")])
            return bool(test_response and test_response.content)
            
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False 

def _debug_prompt_details(full_prompt: str, context_documents: List, question: str) -> None:
    """
    Enhanced debugging function to show comprehensive prompt information.
    
    Args:
        full_prompt: The complete prompt being sent to LLM
        context_documents: List of retrieved documents
        question: User's question
    """
    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(full_prompt) // 4
    
    # Try to get more accurate token count using tiktoken if available
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Close approximation for Llama
        actual_tokens = len(encoding.encode(full_prompt))
    except:
        actual_tokens = estimated_tokens
    
    print("\n" + "="*100)
    print("🔍 DETAILED PROMPT DEBUGGING")
    print("="*100)
    
    # Question details
    print(f"📝 USER QUESTION:")
    print(f"   Length: {len(question)} chars")
    print(f"   Content: {question}")
    print()
    
    # Context documents summary
    print(f"📊 CONTEXT DOCUMENTS: {len(context_documents)} documents")
    if context_documents:
        audio_count = sum(1 for doc in context_documents if doc.metadata.get('modality') == 'audio')
        frame_count = sum(1 for doc in context_documents if doc.metadata.get('modality') == 'frame')
        print(f"   - Audio segments: {audio_count}")
        print(f"   - Frame segments: {frame_count}")
        
        # Show document details
        for i, doc in enumerate(context_documents, 1):
            video_id = doc.metadata.get('video_id', 'unknown')
            start = doc.metadata.get('start', 0)
            end = doc.metadata.get('end', 0)
            modality = doc.metadata.get('modality', 'unknown')
            content_preview = doc.page_content[:50] + "..." if len(doc.page_content) > 50 else doc.page_content
            
            print(f"   {i}. [{video_id}:{start:.1f}-{end:.1f}|{modality}] {content_preview}")
    print()
    
    # Prompt statistics
    print(f"📏 PROMPT STATISTICS:")
    print(f"   - Total length: {len(full_prompt):,} characters")
    print(f"   - Estimated tokens: {estimated_tokens:,}")
    print(f"   - Actual tokens (tiktoken): {actual_tokens:,}")
    print(f"   - Lines: {full_prompt.count(chr(10)) + 1}")
    print()
    
    # Show sections of the prompt
    lines = full_prompt.split('\n')
    
    # Find key sections
    context_start = -1
    answer_start = -1
    
    for i, line in enumerate(lines):
        if "Retrieved Video Segments:" in line:
            context_start = i
        elif line.strip() == "Answer:":
            answer_start = i
    
    print(f"🤖 COMPLETE PROMPT STRUCTURE:")
    print(f"   - System instructions: Lines 1-{context_start if context_start > 0 else 'N/A'}")
    if context_start > 0:
        print(f"   - Context documents: Lines {context_start+1}-{answer_start if answer_start > 0 else len(lines)}")
    print()
    
    print("📤 FULL PROMPT SENT TO LLM:")
    print("-" * 80)
    print(full_prompt)
    print("-" * 80)
    print(f"🎯 Token budget impact: {actual_tokens:,} tokens used")
    print("="*100)
    print() 