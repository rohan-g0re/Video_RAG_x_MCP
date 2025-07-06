"""
MCP Models for Video RAG System.

This module defines data models for MCP (Model Context Protocol) communication,
including request/response structures, model types, and message formats.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

# Base exception for model-related errors
class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelType(Enum):
    """Available AI model types."""
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4_TURBO = "gpt-4-turbo-preview"
    GPT4 = "gpt-4"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet-20241022"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"


class ModelCapability(Enum):
    """Model capabilities."""
    TEXT_GENERATION = "text_generation"
    VISION_ANALYSIS = "vision_analysis"
    CODE_GENERATION = "code_generation"
    REASONING = "reasoning"
    CONVERSATION = "conversation"


class MessageRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MCPMessageType(Enum):
    """MCP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens."""
        return self.prompt_tokens + self.completion_tokens
    
    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'TokenUsage':
        """Create from dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0)
        )


@dataclass
class MCPMessage:
    """Base MCP message structure."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MCPMessageType = field(default=None)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['type'] = self.type.value
        return result


@dataclass
class MCPRequest(MCPMessage):
    """MCP request message."""
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    model: Optional[ModelType] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    timeout: int = 30
    
    def __post_init__(self):
        super().__post_init__()
        self.type = MCPMessageType.REQUEST
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        if self.model:
            result['model'] = self.model.value
        return result


@dataclass
class MCPResponse(MCPMessage):
    """MCP response message."""
    request_id: str = ""
    success: bool = False
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    token_usage: Optional[TokenUsage] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        super().__post_init__()
        self.type = MCPMessageType.RESPONSE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = super().to_dict()
        if self.token_usage:
            result['token_usage'] = self.token_usage.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        """Create from dictionary."""
        token_usage = None
        if 'token_usage' in data:
            token_usage = TokenUsage.from_dict(data.pop('token_usage'))
        
        timestamp = datetime.fromisoformat(data.pop('timestamp'))
        msg_type = MCPMessageType(data.pop('type'))
        
        return cls(
            timestamp=timestamp,
            token_usage=token_usage,
            **data
        )


@dataclass
class VisionAnalysisRequest:
    """Request for vision analysis."""
    image_path: Path = field(default=None)
    prompt: str = ""
    detail_level: str = "high"  # "low", "high", "auto"
    model: ModelType = ModelType.GPT4_VISION
    max_tokens: int = 2000
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.image_path is None:
            return  # Skip validation if no path provided
        
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
        
        if not self.image_path.exists():
            raise ModelError(f"Image file not found: {self.image_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "image_path": str(self.image_path),
            "prompt": self.prompt,
            "detail_level": self.detail_level,
            "model": self.model.value,
            "max_tokens": self.max_tokens,
        }


@dataclass
class TextGenerationRequest:
    """Request for text generation."""
    prompt: str = ""
    model: ModelType = ModelType.CLAUDE_3_5_SONNET
    max_tokens: int = 1000
    temperature: float = 0.7
    system_prompt: Optional[str] = None
    context: Optional[List[Dict[str, str]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "model": self.model.value,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
            "context": self.context or [],
        }


@dataclass
class QueryEnhancementRequest:
    """Request for query enhancement."""
    original_query: str = ""
    context: Optional[str] = None
    model: ModelType = ModelType.GPT4_TURBO
    max_tokens: int = 500
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_query": self.original_query,
            "context": self.context,
            "model": self.model.value,
            "max_tokens": self.max_tokens,
        }


@dataclass
class AnswerGenerationRequest:
    """Request for answer generation."""
    question: str = ""
    context_snippets: List[str] = field(default_factory=list)
    video_metadata: Optional[Dict[str, Any]] = None
    model: ModelType = ModelType.CLAUDE_3_5_SONNET
    max_tokens: int = 1500
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "context_snippets": self.context_snippets,
            "video_metadata": self.video_metadata,
            "model": self.model.value,
            "max_tokens": self.max_tokens,
        }


@dataclass
class ModelUsageStats:
    """Model usage statistics."""
    model: ModelType = field(default=None)
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_response_time: float = 0.0
    last_used: Optional[datetime] = None
    
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def add_request(
        self,
        success: bool,
        tokens: int = 0,
        cost: float = 0.0,
        response_time: float = 0.0
    ):
        """Add a request to statistics."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens += tokens
        self.total_cost += cost
        
        # Update average response time
        if self.total_requests > 1:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time)
                / self.total_requests
            )
        else:
            self.average_response_time = response_time
        
        self.last_used = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model.value,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "average_response_time": self.average_response_time,
            "success_rate": self.success_rate(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }


@dataclass
class MCPConfiguration:
    """MCP client configuration."""
    cursor_pro_endpoint: str = ""
    auth_token: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    connection_pool_size: int = 5
    enable_logging: bool = True
    log_level: str = "INFO"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPConfiguration':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class MCPHealthCheck:
    """MCP service health check result."""
    service: str = ""
    status: str = "unknown"  # "healthy", "degraded", "unhealthy"
    response_time: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status == "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "service": self.service,
            "status": self.status,
            "response_time": self.response_time,
            "last_check": self.last_check.isoformat(),
            "error_message": self.error_message,
            "is_healthy": self.is_healthy(),
        } 