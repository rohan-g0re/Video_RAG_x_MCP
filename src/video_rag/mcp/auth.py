"""
Authentication Module for MCP Integration.

This module handles authentication with Cursor Pro via MCP, including token management,
session handling, and authentication flow for accessing AI models.
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import requests
from cryptography.fernet import Fernet


class AuthError(Exception):
    """Base exception for authentication errors."""
    pass


class AuthTokenExpiredError(AuthError):
    """Exception raised when authentication token has expired."""
    pass


class AuthInvalidCredentialsError(AuthError):
    """Exception raised when credentials are invalid."""
    pass


@dataclass
class AuthToken:
    """Authentication token with metadata."""
    token: str
    token_type: str = "Bearer"
    expires_at: Optional[datetime] = None
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        return datetime.now() >= self.expires_at
    
    def time_until_expiry(self) -> Optional[timedelta]:
        """Get time until token expires."""
        if not self.expires_at:
            return None
        return self.expires_at - datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token": self.token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuthToken':
        """Create from dictionary."""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])
        
        return cls(
            token=data["token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
        )


class MCPAuthenticator(ABC):
    """Abstract base class for MCP authentication."""
    
    @abstractmethod
    async def authenticate(self) -> AuthToken:
        """Authenticate and return auth token."""
        pass
    
    @abstractmethod
    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """Refresh authentication token."""
        pass
    
    @abstractmethod
    async def validate_token(self, token: AuthToken) -> bool:
        """Validate authentication token."""
        pass


class CursorProAuth(MCPAuthenticator):
    """Cursor Pro authentication via MCP."""
    
    def __init__(
        self,
        cursor_pro_endpoint: str,
        credentials_file: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        self.cursor_pro_endpoint = cursor_pro_endpoint
        self.credentials_file = credentials_file or Path.home() / ".cursor" / "credentials.json"
        self.cache_dir = cache_dir or Path.home() / ".video_rag" / "auth"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cipher_suite = self._init_encryption()
        self._cached_token: Optional[AuthToken] = None
    
    def _init_encryption(self) -> Fernet:
        """Initialize encryption for token storage."""
        key_file = self.cache_dir / "auth.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            # Secure the key file
            os.chmod(key_file, 0o600)
        
        return Fernet(key)
    
    def _load_credentials(self) -> Dict[str, str]:
        """Load Cursor Pro credentials."""
        if not self.credentials_file.exists():
            raise AuthError(f"Credentials file not found: {self.credentials_file}")
        
        try:
            with open(self.credentials_file, 'r') as f:
                credentials = json.load(f)
            
            required_fields = ["api_key", "user_id"]
            missing_fields = [field for field in required_fields if field not in credentials]
            
            if missing_fields:
                raise AuthError(f"Missing credentials fields: {missing_fields}")
            
            return credentials
        
        except json.JSONDecodeError as e:
            raise AuthError(f"Invalid credentials file format: {e}")
        except Exception as e:
            raise AuthError(f"Failed to load credentials: {e}")
    
    def _save_token_cache(self, token: AuthToken) -> None:
        """Save token to encrypted cache."""
        try:
            token_data = json.dumps(token.to_dict()).encode()
            encrypted_data = self._cipher_suite.encrypt(token_data)
            
            cache_file = self.cache_dir / "token.cache"
            with open(cache_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Secure the cache file
            os.chmod(cache_file, 0o600)
            
        except Exception as e:
            self.logger.warning(f"Failed to cache token: {e}")
    
    def _load_token_cache(self) -> Optional[AuthToken]:
        """Load token from encrypted cache."""
        try:
            cache_file = self.cache_dir / "token.cache"
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._cipher_suite.decrypt(encrypted_data)
            token_data = json.loads(decrypted_data.decode())
            
            token = AuthToken.from_dict(token_data)
            
            # Check if token is still valid
            if token.is_expired():
                self.logger.info("Cached token is expired")
                return None
            
            return token
        
        except Exception as e:
            self.logger.warning(f"Failed to load cached token: {e}")
            return None
    
    async def authenticate(self) -> AuthToken:
        """Authenticate with Cursor Pro."""
        # First, try to use cached token
        cached_token = self._load_token_cache()
        if cached_token and not cached_token.is_expired():
            self.logger.info("Using cached authentication token")
            self._cached_token = cached_token
            return cached_token
        
        # If cached token is expired, try to refresh it
        if cached_token and cached_token.refresh_token:
            try:
                refreshed_token = await self.refresh_token(cached_token.refresh_token)
                self._cached_token = refreshed_token
                return refreshed_token
            except AuthError:
                self.logger.info("Token refresh failed, proceeding with full authentication")
        
        # Perform full authentication
        credentials = self._load_credentials()
        
        try:
            auth_response = await self._perform_authentication(credentials)
            token = self._parse_auth_response(auth_response)
            
            # Cache the new token
            self._save_token_cache(token)
            self._cached_token = token
            
            self.logger.info("Authentication successful")
            return token
        
        except Exception as e:
            raise AuthError(f"Authentication failed: {e}")
    
    async def _perform_authentication(self, credentials: Dict[str, str]) -> Dict[str, Any]:
        """Perform the actual authentication request."""
        auth_url = f"{self.cursor_pro_endpoint}/auth/login"
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VideoRAG-MCP/1.0",
        }
        
        payload = {
            "api_key": credentials["api_key"],
            "user_id": credentials["user_id"],
            "scope": "model_access vision_analysis text_generation",
        }
        
        try:
            response = requests.post(
                auth_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 401:
                raise AuthInvalidCredentialsError("Invalid credentials")
            elif response.status_code != 200:
                raise AuthError(f"Authentication request failed: {response.status_code}")
            
            return response.json()
        
        except requests.RequestException as e:
            raise AuthError(f"Authentication request failed: {e}")
    
    def _parse_auth_response(self, response: Dict[str, Any]) -> AuthToken:
        """Parse authentication response."""
        try:
            access_token = response["access_token"]
            token_type = response.get("token_type", "Bearer")
            expires_in = response.get("expires_in")
            refresh_token = response.get("refresh_token")
            scope = response.get("scope")
            
            expires_at = None
            if expires_in:
                expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            return AuthToken(
                token=access_token,
                token_type=token_type,
                expires_at=expires_at,
                refresh_token=refresh_token,
                scope=scope,
            )
        
        except KeyError as e:
            raise AuthError(f"Invalid authentication response: missing {e}")
    
    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """Refresh authentication token."""
        refresh_url = f"{self.cursor_pro_endpoint}/auth/refresh"
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "VideoRAG-MCP/1.0",
        }
        
        payload = {
            "refresh_token": refresh_token,
        }
        
        try:
            response = requests.post(
                refresh_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 401:
                raise AuthTokenExpiredError("Refresh token expired")
            elif response.status_code != 200:
                raise AuthError(f"Token refresh failed: {response.status_code}")
            
            auth_response = response.json()
            token = self._parse_auth_response(auth_response)
            
            # Cache the refreshed token
            self._save_token_cache(token)
            
            self.logger.info("Token refresh successful")
            return token
        
        except requests.RequestException as e:
            raise AuthError(f"Token refresh request failed: {e}")
    
    async def validate_token(self, token: AuthToken) -> bool:
        """Validate authentication token."""
        if token.is_expired():
            return False
        
        validate_url = f"{self.cursor_pro_endpoint}/auth/validate"
        
        headers = {
            "Authorization": f"{token.token_type} {token.token}",
            "User-Agent": "VideoRAG-MCP/1.0",
        }
        
        try:
            response = requests.get(
                validate_url,
                headers=headers,
                timeout=10
            )
            
            return response.status_code == 200
        
        except requests.RequestException:
            return False
    
    async def get_valid_token(self) -> AuthToken:
        """Get a valid authentication token, refreshing if necessary."""
        if self._cached_token and not self._cached_token.is_expired():
            return self._cached_token
        
        # Token is expired or doesn't exist, authenticate
        return await self.authenticate()
    
    def clear_cache(self) -> None:
        """Clear cached authentication data."""
        try:
            cache_file = self.cache_dir / "token.cache"
            if cache_file.exists():
                cache_file.unlink()
            
            self._cached_token = None
            self.logger.info("Authentication cache cleared")
        
        except Exception as e:
            self.logger.warning(f"Failed to clear cache: {e}")


class MockAuthenticator(MCPAuthenticator):
    """Mock authenticator for testing."""
    
    def __init__(self, should_succeed: bool = True):
        self.should_succeed = should_succeed
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def authenticate(self) -> AuthToken:
        """Mock authentication."""
        if not self.should_succeed:
            raise AuthError("Mock authentication failed")
        
        return AuthToken(
            token="mock_token_12345",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token="mock_refresh_token",
            scope="model_access vision_analysis text_generation",
        )
    
    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """Mock token refresh."""
        if not self.should_succeed:
            raise AuthTokenExpiredError("Mock refresh failed")
        
        return AuthToken(
            token="mock_refreshed_token_67890",
            token_type="Bearer",
            expires_at=datetime.now() + timedelta(hours=1),
            refresh_token="mock_new_refresh_token",
            scope="model_access vision_analysis text_generation",
        )
    
    async def validate_token(self, token: AuthToken) -> bool:
        """Mock token validation."""
        return self.should_succeed and not token.is_expired() 