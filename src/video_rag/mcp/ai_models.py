"""
AI Model Interfaces for MCP Video RAG System.

This module provides abstract interfaces and concrete implementations for different
AI models accessible via Cursor Pro, including vision analysis, text generation,
and query enhancement capabilities.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path

from .client import MCPClient
from .models import (
    ModelType, ModelCapability, VisionAnalysisRequest,
    TextGenerationRequest, QueryEnhancementRequest, AnswerGenerationRequest
)


class AIModelInterface(ABC):
    """Abstract base interface for AI models."""
    
    @abstractmethod
    def get_model_type(self) -> ModelType:
        """Get the model type."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[ModelCapability]:
        """Get model capabilities."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if model is available."""
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum token limit for this model."""
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> float:
        """Get cost per token (for usage tracking)."""
        pass


class VisionModel(AIModelInterface):
    """Abstract interface for vision analysis models."""
    
    @abstractmethod
    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        detail_level: str = "high"
    ) -> str:
        """Analyze image and return description."""
        pass
    
    @abstractmethod
    async def analyze_video_frame(
        self,
        frame_path: Path,
        context: str,
        timestamp: float
    ) -> str:
        """Analyze video frame with temporal context."""
        pass
    
    @abstractmethod
    async def compare_frames(
        self,
        frame1_path: Path,
        frame2_path: Path,
        comparison_prompt: str
    ) -> str:
        """Compare two frames and describe differences."""
        pass


class LanguageModel(AIModelInterface):
    """Abstract interface for language models."""
    
    @abstractmethod
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def generate_answer(
        self,
        question: str,
        context: List[str],
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate answer based on context."""
        pass
    
    @abstractmethod
    async def summarize_content(
        self,
        content: str,
        max_length: int = 200
    ) -> str:
        """Summarize content."""
        pass


class QueryModel(AIModelInterface):
    """Abstract interface for query enhancement models."""
    
    @abstractmethod
    async def enhance_query(
        self,
        original_query: str,
        context: Optional[str] = None
    ) -> str:
        """Enhance query for better search results."""
        pass
    
    @abstractmethod
    async def extract_keywords(
        self,
        query: str
    ) -> List[str]:
        """Extract keywords from query."""
        pass
    
    @abstractmethod
    async def classify_query_intent(
        self,
        query: str
    ) -> Dict[str, float]:
        """Classify query intent with confidence scores."""
        pass


class GPT4VisionModel(VisionModel):
    """GPT-4V implementation for vision analysis."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_model_type(self) -> ModelType:
        """Get the model type."""
        return ModelType.GPT4_VISION
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get model capabilities."""
        return [
            ModelCapability.VISION_ANALYSIS,
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING
        ]
    
    async def is_available(self) -> bool:
        """Check if model is available."""
        try:
            health = await self.mcp_client.health_check()
            return health.is_healthy()
        except Exception:
            return False
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return 4096
    
    def get_cost_per_token(self) -> float:
        """Get cost per token."""
        return 0.0  # Using Cursor Pro subscription
    
    async def analyze_image(
        self,
        image_path: Path,
        prompt: str,
        detail_level: str = "high"
    ) -> str:
        """Analyze image using GPT-4V."""
        request = VisionAnalysisRequest(
            image_path=image_path,
            prompt=prompt,
            detail_level=detail_level,
            model=self.get_model_type()
        )
        
        return await self.mcp_client.analyze_vision(request)
    
    async def analyze_video_frame(
        self,
        frame_path: Path,
        context: str,
        timestamp: float
    ) -> str:
        """Analyze video frame with temporal context."""
        prompt = f"""
        Analyze this video frame captured at timestamp {timestamp:.2f} seconds.
        
        Context: {context}
        
        Please describe:
        1. What is happening in this frame
        2. Key objects, people, or elements visible
        3. Actions or movements taking place
        4. Relevant visual details for video search and retrieval
        5. Any text or readable content in the frame
        
        Provide a detailed but concise description that would help someone find this moment in the video.
        """
        
        return await self.analyze_image(frame_path, prompt, "high")
    
    async def compare_frames(
        self,
        frame1_path: Path,
        frame2_path: Path,
        comparison_prompt: str
    ) -> str:
        """Compare two frames and describe differences."""
        # For simplicity, analyze each frame separately and note comparison
        # In a full implementation, you'd send both images in one request
        frame1_desc = await self.analyze_image(frame1_path, f"Describe this frame: {comparison_prompt}")
        frame2_desc = await self.analyze_image(frame2_path, f"Describe this frame: {comparison_prompt}")
        
        return f"Frame 1: {frame1_desc}\n\nFrame 2: {frame2_desc}"


class GPT4LanguageModel(LanguageModel, QueryModel):
    """GPT-4 implementation for text generation and query enhancement."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_model_type(self) -> ModelType:
        """Get the model type."""
        return ModelType.GPT4_TURBO
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get model capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CONVERSATION
        ]
    
    async def is_available(self) -> bool:
        """Check if model is available."""
        try:
            health = await self.mcp_client.health_check()
            return health.is_healthy()
        except Exception:
            return False
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return 128000
    
    def get_cost_per_token(self) -> float:
        """Get cost per token."""
        return 0.0  # Using Cursor Pro subscription
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text using GPT-4."""
        request = TextGenerationRequest(
            prompt=prompt,
            model=self.get_model_type(),
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        return await self.mcp_client.generate_text(request)
    
    async def generate_answer(
        self,
        question: str,
        context: List[str],
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate answer based on context."""
        request = AnswerGenerationRequest(
            question=question,
            context_snippets=context,
            video_metadata=video_metadata,
            model=self.get_model_type()
        )
        
        return await self.mcp_client.generate_answer(request)
    
    async def summarize_content(
        self,
        content: str,
        max_length: int = 200
    ) -> str:
        """Summarize content."""
        prompt = f"""
        Please summarize the following content in approximately {max_length} words:
        
        {content}
        
        Summary:
        """
        
        return await self.generate_text(prompt, max_tokens=max_length * 2)
    
    async def enhance_query(
        self,
        original_query: str,
        context: Optional[str] = None
    ) -> str:
        """Enhance query for better search results."""
        request = QueryEnhancementRequest(
            original_query=original_query,
            context=context,
            model=self.get_model_type()
        )
        
        return await self.mcp_client.enhance_query(request)
    
    async def extract_keywords(
        self,
        query: str
    ) -> List[str]:
        """Extract keywords from query."""
        prompt = f"""
        Extract the most important keywords from this query for video search:
        
        Query: {query}
        
        Return only the keywords, separated by commas, without any other text.
        """
        
        result = await self.generate_text(prompt, max_tokens=100, temperature=0.3)
        keywords = [kw.strip() for kw in result.split(',')]
        return [kw for kw in keywords if kw]
    
    async def classify_query_intent(
        self,
        query: str
    ) -> Dict[str, float]:
        """Classify query intent with confidence scores."""
        prompt = f"""
        Classify the intent of this video search query into categories with confidence scores (0-1):
        
        Query: {query}
        
        Categories:
        - factual_search: Looking for specific information or facts
        - moment_search: Looking for a specific moment or scene
        - person_search: Looking for a specific person
        - action_search: Looking for specific actions or activities
        - object_search: Looking for specific objects
        - emotional_search: Looking for emotional content or reactions
        
        Respond in JSON format: {{"category": confidence, ...}}
        """
        
        result = await self.generate_text(prompt, max_tokens=200, temperature=0.3)
        
        try:
            import json
            return json.loads(result)
        except:
            # Fallback if JSON parsing fails
            return {"general_search": 1.0}


class ClaudeLanguageModel(LanguageModel):
    """Claude 3.5 Sonnet implementation for advanced text generation."""
    
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def get_model_type(self) -> ModelType:
        """Get the model type."""
        return ModelType.CLAUDE_3_5_SONNET
    
    def get_capabilities(self) -> List[ModelCapability]:
        """Get model capabilities."""
        return [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.REASONING,
            ModelCapability.CONVERSATION,
            ModelCapability.CODE_GENERATION
        ]
    
    async def is_available(self) -> bool:
        """Check if model is available."""
        try:
            health = await self.mcp_client.health_check()
            return health.is_healthy()
        except Exception:
            return False
    
    def get_max_tokens(self) -> int:
        """Get maximum token limit."""
        return 200000
    
    def get_cost_per_token(self) -> float:
        """Get cost per token."""
        return 0.0  # Using Cursor Pro subscription
    
    async def generate_text(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text using Claude."""
        request = TextGenerationRequest(
            prompt=prompt,
            model=self.get_model_type(),
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        return await self.mcp_client.generate_text(request)
    
    async def generate_answer(
        self,
        question: str,
        context: List[str],
        video_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate comprehensive answer based on context."""
        request = AnswerGenerationRequest(
            question=question,
            context_snippets=context,
            video_metadata=video_metadata,
            model=self.get_model_type(),
            max_tokens=2000  # Claude can handle longer responses
        )
        
        return await self.mcp_client.generate_answer(request)
    
    async def summarize_content(
        self,
        content: str,
        max_length: int = 200
    ) -> str:
        """Summarize content with high quality."""
        system_prompt = """You are an expert at creating concise, informative summaries. 
        Focus on the most important points and maintain clarity."""
        
        prompt = f"""
        Summarize the following content in approximately {max_length} words, 
        capturing the key points and main themes:
        
        {content}
        """
        
        return await self.generate_text(
            prompt, 
            max_tokens=max_length * 2, 
            temperature=0.3,
            system_prompt=system_prompt
        )


@dataclass
class ModelSelectionCriteria:
    """Criteria for selecting the best model for a task."""
    task_type: str
    content_length: int
    quality_preference: str = "balanced"  # "speed", "balanced", "quality"
    cost_sensitive: bool = False


class ModelManager:
    """Manager for coordinating multiple AI models."""
    
    def __init__(self, mcp_client: Optional[MCPClient] = None):
        """Initialize ModelManager with optional MCP client."""
        self.mcp_client = mcp_client
        
        # Initialize models only if client is provided
        if mcp_client:
            self.vision_model = GPT4VisionModel(mcp_client)
            self.gpt4_model = GPT4LanguageModel(mcp_client)
            self.claude_model = ClaudeLanguageModel(mcp_client)
        else:
            self.vision_model = None
            self.gpt4_model = None
            self.claude_model = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_available_models(self) -> Dict[str, bool]:
        """Get availability status of all models."""
        models = {
            "gpt4_vision": self.vision_model,
            "gpt4_language": self.gpt4_model,
            "claude_language": self.claude_model,
        }
        
        availability = {}
        for name, model in models.items():
            availability[name] = await model.is_available()
        
        return availability
    
    def select_language_model(self, criteria: ModelSelectionCriteria) -> LanguageModel:
        """Select the best language model based on criteria."""
        if criteria.quality_preference == "quality":
            return self.claude_model
        elif criteria.quality_preference == "speed":
            return self.gpt4_model
        else:  # balanced
            # For balanced, choose based on content length
            if criteria.content_length > 10000:
                return self.claude_model  # Better for long content
            else:
                return self.gpt4_model  # Faster for shorter content
    
    async def analyze_video_frame(
        self,
        frame_path: Path,
        context: str,
        timestamp: float
    ) -> str:
        """Analyze video frame using the vision model."""
        return await self.vision_model.analyze_video_frame(frame_path, context, timestamp)
    
    async def enhance_search_query(
        self,
        original_query: str,
        context: Optional[str] = None
    ) -> str:
        """Enhance search query using the query model."""
        return await self.gpt4_model.enhance_query(original_query, context)
    
    async def generate_comprehensive_answer(
        self,
        question: str,
        context: List[str],
        video_metadata: Optional[Dict[str, Any]] = None,
        quality_preference: str = "quality"
    ) -> str:
        """Generate comprehensive answer using the best available model."""
        criteria = ModelSelectionCriteria(
            task_type="answer_generation",
            content_length=sum(len(ctx) for ctx in context),
            quality_preference=quality_preference
        )
        
        model = self.select_language_model(criteria)
        return await model.generate_answer(question, context, video_metadata)
    
    async def parallel_analysis(
        self,
        frames: List[Path],
        context: str,
        timestamps: List[float]
    ) -> List[str]:
        """Analyze multiple frames in parallel."""
        tasks = [
            self.vision_model.analyze_video_frame(frame, context, timestamp)
            for frame, timestamp in zip(frames, timestamps)
        ]
        
        return await asyncio.gather(*tasks)
    
    async def get_model_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for all models."""
        return {
            "usage_stats": self.mcp_client.get_usage_stats(),
            "available_models": await self.get_available_models(),
            "model_capabilities": {
                "vision": self.vision_model.get_capabilities(),
                "gpt4": self.gpt4_model.get_capabilities(),
                "claude": self.claude_model.get_capabilities(),
            }
        } 