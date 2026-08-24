"""
Rate Limiting Middleware
Protect API endpoints from abuse.
"""
from app.core.time import IST, now_ist
from typing import Optional, Dict
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10  # Max concurrent requests
    
    # Endpoint-specific limits
    inference_per_minute: int = 100
    training_per_hour: int = 10


class RateLimiter:
    """
    Token bucket rate limiter.
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self._requests: Dict[str, list] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Try to get user ID from auth, fallback to IP
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"
        
        # Get IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        
        return f"ip:{request.client.host if request.client else 'unknown'}"
    
    def _get_limit(self, request: Request) -> int:
        """Get rate limit for endpoint."""
        path = request.url.path
        
        if "/inference" in path or "/predict" in path:
            return self.config.inference_per_minute
        
        if "/training" in path:
            return self.config.training_per_hour
        
        return self.config.requests_per_minute
    
    async def check_rate_limit(self, request: Request) -> bool:
        """
        Check if request is within rate limits.
        
        Returns True if allowed, raises HTTPException if rate limited.
        """
        client_id = self._get_client_id(request)
        limit = self._get_limit(request)
        window = timedelta(minutes=1)
        
        async with self._locks[client_id]:
            now = now_ist()
            cutoff = now - window
            
            # Clean old entries
            self._requests[client_id] = [
                ts for ts in self._requests[client_id]
                if ts > cutoff
            ]
            
            # Check limit
            if len(self._requests[client_id]) >= limit:
                retry_after = int((self._requests[client_id][0] + window - now).total_seconds())
                
                logger.warning(f"Rate limit exceeded for {client_id}")
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": limit,
                        "window": "1 minute",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            
            # Add request
            self._requests[client_id].append(now)
        
        return True
    
    def get_usage(self, request: Request) -> Dict:
        """Get current rate limit usage for client."""
        client_id = self._get_client_id(request)
        limit = self._get_limit(request)
        
        now = now_ist()
        cutoff = now - timedelta(minutes=1)
        
        current = len([
            ts for ts in self._requests.get(client_id, [])
            if ts > cutoff
        ])
        
        return {
            "limit": limit,
            "remaining": max(0, limit - current),
            "reset": int((cutoff + timedelta(minutes=1)).timestamp()),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI rate limiting middleware."""
    
    def __init__(self, app, config: RateLimitConfig = None):
        super().__init__(app)
        self.rate_limiter = RateLimiter(config)
        self._excluded_paths = {"/health", "/docs", "/openapi.json", "/redoc"}
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for excluded paths
        if request.url.path in self._excluded_paths:
            return await call_next(request)
        
        # Check rate limit
        await self.rate_limiter.check_rate_limit(request)
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        usage = self.rate_limiter.get_usage(request)
        response.headers["X-RateLimit-Limit"] = str(usage["limit"])
        response.headers["X-RateLimit-Remaining"] = str(usage["remaining"])
        response.headers["X-RateLimit-Reset"] = str(usage["reset"])
        
        return response


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
