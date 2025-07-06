"""
MCP Bridge Implementation for Video RAG System.

This module provides the Model Context Protocol (MCP) bridge layer that connects
the local Video RAG system to Cursor Pro's AI models (GPT-4V, GPT-4, Claude 3.5)
with zero additional cost by leveraging existing Cursor Pro subscriptions.

Key Features:
- MCP Client for communication with Cursor Pro
- Authentication and session management
- Connection pooling and rate limiting
- Retry logic and error handling
- Model abstraction for GPT-4V, GPT-4, and Claude 3.5
- Prompt templates and token optimization
"""

# Import MCP client and connection components
from .client import (
    MCPClient,
    MCPConnection,
    MCPConnectionPool,
    MCPError,
    MCPAuthenticationError,
    MCPRateLimitError,
    MCPTimeoutError,
)

# Import authentication components
from .auth import (
    MCPAuthenticator,
    CursorProAuth,
    AuthToken,
    AuthError,
)

# Import model interfaces
from .models import (
    MCPRequest,
    MCPResponse,
    MCPMessage,
    ModelType,
    ModelCapability,
    TokenUsage,
    ModelError,
)

# Import AI model abstractions
from .ai_models import (
    AIModelInterface,
    VisionModel,
    LanguageModel,
    QueryModel,
    GPT4VisionModel,
    GPT4LanguageModel,
    ClaudeLanguageModel,
    ModelManager,
)

# Import prompt management
from .prompts import (
    PromptTemplate,
    PromptManager,
    VisionPrompts,
    QueryPrompts,
    AnswerPrompts,
)

# Export all public components
__all__ = [
    # Core MCP components
    "MCPClient",
    "MCPConnection", 
    "MCPConnectionPool",
    "MCPError",
    "MCPAuthenticationError",
    "MCPRateLimitError",
    "MCPTimeoutError",
    
    # Authentication
    "MCPAuthenticator",
    "CursorProAuth",
    "AuthToken",
    "AuthError",
    
    # Models and messages
    "MCPRequest",
    "MCPResponse",
    "MCPMessage",
    "ModelType",
    "ModelCapability",
    "TokenUsage",
    "ModelError",
    
    # AI model interfaces
    "AIModelInterface",
    "VisionModel",
    "LanguageModel", 
    "QueryModel",
    "GPT4VisionModel",
    "GPT4LanguageModel",
    "ClaudeLanguageModel",
    "ModelManager",
    
    # Prompt management
    "PromptTemplate",
    "PromptManager",
    "VisionPrompts",
    "QueryPrompts",
    "AnswerPrompts",
] 