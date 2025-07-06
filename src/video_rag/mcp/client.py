"""
MCP Client for Video RAG System.

This module provides the core MCP client that handles communication with Cursor Pro
models, including connection management, rate limiting, retry logic, and error handling.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, AsyncGenerator
from urllib.parse import urljoin

import aiohttp
import asyncio
from collections import deque
from threading import Lock

from .auth import MCPAuthenticator, AuthToken, AuthError
from .models import (
    MCPRequest, MCPResponse, MCPConfiguration, MCPHealthCheck,
    ModelType, TokenUsage, VisionAnalysisRequest, TextGenerationRequest,
    QueryEnhancementRequest, AnswerGenerationRequest, ModelUsageStats
)


class MCPError(Exception):
    """Base exception for MCP operations."""
    pass


class MCPAuthenticationError(MCPError):
    """Exception raised for authentication errors."""
    pass


class MCPRateLimitError(MCPError):
    """Exception raised when rate limit is exceeded."""
    pass


class MCPTimeoutError(MCPError):
    """Exception raised for timeout errors."""
    pass


class MCPModelError(MCPError):
    """Exception raised for model-specific errors."""
    pass


@dataclass
class RateLimiter:
    """Rate limiter for MCP requests."""
    max_requests: int
    window_seconds: int
    _requests: deque = field(default_factory=deque)
    _lock: Lock = field(default_factory=Lock)
    
    def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding rate limit."""
        with self._lock:
            now = time.time()
            # Remove old requests outside the window
            while self._requests and self._requests[0] <= now - self.window_seconds:
                self._requests.popleft()
            
            return len(self._requests) < self.max_requests
    
    def record_request(self) -> None:
        """Record a new request."""
        with self._lock:
            self._requests.append(time.time())
    
    def time_until_next_request(self) -> float:
        """Get time until next request can be made."""
        with self._lock:
            if len(self._requests) < self.max_requests:
                return 0.0
            
            oldest_request = self._requests[0]
            return max(0.0, self.window_seconds - (time.time() - oldest_request))


class MCPConnection:
    """Single MCP connection with session management."""
    
    def __init__(
        self,
        endpoint: str,
        authenticator: MCPAuthenticator,
        timeout: int = 30
    ):
        self.endpoint = endpoint
        self.authenticator = authenticator
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.auth_token: Optional[AuthToken] = None
        self.last_used = datetime.now()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect(self) -> None:
        """Establish connection."""
        if self.session and not self.session.closed:
            return
        
        # Create new session
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "VideoRAG-MCP/1.0"}
        )
        
        # Authenticate
        self.auth_token = await self.authenticator.authenticate()
        self.last_used = datetime.now()
        
        self.logger.info("MCP connection established")
    
    async def disconnect(self) -> None:
        """Close connection."""
        if self.session and not self.session.closed:
            await self.session.close()
        
        self.session = None
        self.auth_token = None
        
        self.logger.info("MCP connection closed")
    
    async def ensure_connected(self) -> None:
        """Ensure connection is active and authenticated."""
        if not self.session or self.session.closed:
            await self.connect()
            return
        
        # Check if token needs refresh
        if self.auth_token and self.auth_token.is_expired():
            try:
                self.auth_token = await self.authenticator.get_valid_token()
            except AuthError as e:
                self.logger.error(f"Failed to refresh token: {e}")
                await self.connect()
        
        self.last_used = datetime.now()
    
    async def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with authentication."""
        await self.ensure_connected()
        
        if not self.auth_token:
            raise MCPAuthenticationError("No valid authentication token")
        
        request_headers = {
            "Authorization": f"{self.auth_token.token_type} {self.auth_token.token}",
            "Content-Type": "application/json",
        }
        
        if headers:
            request_headers.update(headers)
        
        url = urljoin(self.endpoint, endpoint)
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                headers=request_headers
            ) as response:
                
                if response.status == 401:
                    raise MCPAuthenticationError("Authentication failed")
                elif response.status == 429:
                    raise MCPRateLimitError("Rate limit exceeded")
                elif response.status >= 400:
                    error_text = await response.text()
                    raise MCPError(f"Request failed: {response.status} - {error_text}")
                
                return await response.json()
        
        except asyncio.TimeoutError:
            raise MCPTimeoutError("Request timed out")
        except aiohttp.ClientError as e:
            raise MCPError(f"Request failed: {e}")
    
    def is_idle(self, max_idle_time: int = 300) -> bool:
        """Check if connection has been idle too long."""
        idle_time = datetime.now() - self.last_used
        return idle_time.total_seconds() > max_idle_time


class MCPConnectionPool:
    """Connection pool for managing multiple MCP connections."""
    
    def __init__(
        self,
        endpoint: str,
        authenticator: MCPAuthenticator,
        pool_size: int = 5,
        timeout: int = 30
    ):
        self.endpoint = endpoint
        self.authenticator = authenticator
        self.pool_size = pool_size
        self.timeout = timeout
        
        self._pool: List[MCPConnection] = []
        self._lock = asyncio.Lock()
        self._created_connections = 0
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def get_connection(self) -> MCPConnection:
        """Get a connection from the pool."""
        async with self._lock:
            # Try to find an available connection
            for connection in self._pool:
                if not connection.session or connection.session.closed:
                    continue
                
                # Return the connection
                self._pool.remove(connection)
                return connection
            
            # Create new connection if pool not at capacity
            if self._created_connections < self.pool_size:
                connection = MCPConnection(
                    self.endpoint,
                    self.authenticator,
                    self.timeout
                )
                self._created_connections += 1
                self.logger.debug(f"Created new connection ({self._created_connections}/{self.pool_size})")
                return connection
            
            # Wait for a connection to become available
            raise MCPError("Connection pool exhausted")
    
    async def return_connection(self, connection: MCPConnection) -> None:
        """Return a connection to the pool."""
        async with self._lock:
            if len(self._pool) < self.pool_size and not connection.is_idle():
                self._pool.append(connection)
            else:
                await connection.disconnect()
                self._created_connections -= 1
    
    async def cleanup_idle_connections(self) -> None:
        """Clean up idle connections."""
        async with self._lock:
            active_connections = []
            
            for connection in self._pool:
                if connection.is_idle():
                    await connection.disconnect()
                    self._created_connections -= 1
                    self.logger.debug("Closed idle connection")
                else:
                    active_connections.append(connection)
            
            self._pool = active_connections
    
    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            for connection in self._pool:
                await connection.disconnect()
            
            self._pool.clear()
            self._created_connections = 0
            
            self.logger.info("All pool connections closed")


class MCPClient:
    """Main MCP client for communicating with Cursor Pro models."""
    
    def __init__(
        self,
        config: MCPConfiguration,
        authenticator: MCPAuthenticator
    ):
        self.config = config
        self.authenticator = authenticator
        
        # Initialize connection pool
        self.connection_pool = MCPConnectionPool(
            endpoint=config.cursor_pro_endpoint,
            authenticator=authenticator,
            pool_size=config.connection_pool_size,
            timeout=config.timeout
        )
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            max_requests=config.rate_limit_requests,
            window_seconds=config.rate_limit_window
        )
        
        # Usage statistics
        self.usage_stats: Dict[ModelType, ModelUsageStats] = {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Close the MCP client."""
        await self.connection_pool.close_all()
        self.logger.info("MCP client closed")
    
    async def _make_request_with_retry(
        self,
        request: MCPRequest,
        endpoint: str
    ) -> MCPResponse:
        """Make request with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Rate limiting
                if not self.rate_limiter.can_make_request():
                    wait_time = self.rate_limiter.time_until_next_request()
                    if wait_time > 0:
                        self.logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                
                self.rate_limiter.record_request()
                
                # Get connection from pool
                connection = await self.connection_pool.get_connection()
                
                try:
                    start_time = time.time()
                    
                    # Make the request
                    response_data = await connection.make_request(
                        method="POST",
                        endpoint=endpoint,
                        data=request.to_dict()
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Parse response
                    response = MCPResponse(
                        id="",
                        request_id=request.id,
                        success=response_data.get("success", True),
                        result=response_data.get("result"),
                        error=response_data.get("error"),
                        error_code=response_data.get("error_code"),
                        processing_time=processing_time
                    )
                    
                    # Parse token usage if present
                    if "usage" in response_data:
                        usage_data = response_data["usage"]
                        response.token_usage = TokenUsage(
                            prompt_tokens=usage_data.get("prompt_tokens", 0),
                            completion_tokens=usage_data.get("completion_tokens", 0),
                            total_tokens=usage_data.get("total_tokens", 0)
                        )
                    
                    # Update usage statistics
                    if request.model:
                        self._update_usage_stats(
                            request.model,
                            response.success,
                            response.token_usage.total_tokens if response.token_usage else 0,
                            processing_time
                        )
                    
                    return response
                
                finally:
                    await self.connection_pool.return_connection(connection)
            
            except (MCPTimeoutError, MCPRateLimitError, MCPError) as e:
                last_exception = e
                
                if attempt < self.config.max_retries:
                    wait_time = self.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(f"Request failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Request failed after {self.config.max_retries} retries: {e}")
                    break
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise MCPError("Request failed without specific error")
    
    def _update_usage_stats(
        self,
        model: ModelType,
        success: bool,
        tokens: int,
        response_time: float
    ) -> None:
        """Update model usage statistics."""
        if model not in self.usage_stats:
            self.usage_stats[model] = ModelUsageStats(model=model)
        
        self.usage_stats[model].add_request(
            success=success,
            tokens=tokens,
            response_time=response_time
        )
    
    async def analyze_vision(self, request: VisionAnalysisRequest) -> str:
        """Analyze image using vision model."""
        mcp_request = MCPRequest(
            id="",
            method="vision.analyze",
            params=request.to_dict(),
            model=request.model,
            max_tokens=request.max_tokens
        )
        
        response = await self._make_request_with_retry(mcp_request, "/vision/analyze")
        
        if not response.success:
            raise MCPModelError(f"Vision analysis failed: {response.error}")
        
        return response.result.get("description", "")
    
    async def generate_text(self, request: TextGenerationRequest) -> str:
        """Generate text using language model."""
        mcp_request = MCPRequest(
            id="",
            method="text.generate",
            params=request.to_dict(),
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        response = await self._make_request_with_retry(mcp_request, "/text/generate")
        
        if not response.success:
            raise MCPModelError(f"Text generation failed: {response.error}")
        
        return response.result.get("text", "")
    
    async def enhance_query(self, request: QueryEnhancementRequest) -> str:
        """Enhance query using language model."""
        mcp_request = MCPRequest(
            id="",
            method="query.enhance",
            params=request.to_dict(),
            model=request.model,
            max_tokens=request.max_tokens
        )
        
        response = await self._make_request_with_retry(mcp_request, "/query/enhance")
        
        if not response.success:
            raise MCPModelError(f"Query enhancement failed: {response.error}")
        
        return response.result.get("enhanced_query", request.original_query)
    
    async def generate_answer(self, request: AnswerGenerationRequest) -> str:
        """Generate answer using language model."""
        mcp_request = MCPRequest(
            id="",
            method="answer.generate",
            params=request.to_dict(),
            model=request.model,
            max_tokens=request.max_tokens
        )
        
        response = await self._make_request_with_retry(mcp_request, "/answer/generate")
        
        if not response.success:
            raise MCPModelError(f"Answer generation failed: {response.error}")
        
        return response.result.get("answer", "")
    
    async def health_check(self) -> MCPHealthCheck:
        """Perform health check on MCP service."""
        start_time = time.time()
        
        try:
            connection = await self.connection_pool.get_connection()
            
            try:
                await connection.make_request("GET", "/health")
                response_time = time.time() - start_time
                
                return MCPHealthCheck(
                    service="cursor_pro_mcp",
                    status="healthy",
                    response_time=response_time,
                    last_check=datetime.now()
                )
            
            finally:
                await self.connection_pool.return_connection(connection)
        
        except Exception as e:
            response_time = time.time() - start_time
            
            return MCPHealthCheck(
                service="cursor_pro_mcp",
                status="unhealthy",
                response_time=response_time,
                last_check=datetime.now(),
                error_message=str(e)
            )
    
    def get_usage_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get usage statistics for all models."""
        return {
            model.value: stats.to_dict()
            for model, stats in self.usage_stats.items()
        }
    
    async def cleanup(self) -> None:
        """Perform cleanup tasks."""
        await self.connection_pool.cleanup_idle_connections()
        self.logger.debug("Cleanup completed") 