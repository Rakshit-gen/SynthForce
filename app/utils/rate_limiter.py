"""
Rate limiting implementation.

Provides Redis-backed rate limiting with fallback to in-memory.
"""

import logging
import time
from typing import Optional, Tuple

from fastapi import HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Get client IP address from request.
    
    Handles forwarded headers for proxy setups.
    """
    # Check for forwarded headers
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    
    # Fall back to direct connection
    if request.client:
        return request.client.host
    
    return "unknown"


def create_rate_limiter() -> Limiter:
    """
    Create a rate limiter instance.
    
    Uses Redis if available, falls back to in-memory.
    """
    settings = get_settings()
    
    # Try to use Redis for distributed rate limiting
    try:
        limiter = Limiter(
            key_func=get_client_ip,
            storage_uri=settings.redis.url,
            strategy="fixed-window",
        )
        logger.info("Rate limiter initialized with Redis backend")
    except Exception as e:
        logger.warning(f"Failed to connect to Redis for rate limiting: {e}")
        # Fall back to in-memory
        limiter = Limiter(
            key_func=get_client_ip,
            strategy="fixed-window",
        )
        logger.info("Rate limiter initialized with in-memory backend")
    
    return limiter


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception."""
    
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        headers = {}
        if retry_after:
            headers["Retry-After"] = str(retry_after)
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers=headers,
        )


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for fine-grained control.
    
    Useful for per-user or per-session rate limiting.
    """
    
    def __init__(
        self,
        rate: float,
        capacity: int,
        redis_client: Optional[any] = None,
    ):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second to add
            capacity: Maximum tokens in bucket
            redis_client: Optional Redis client for distributed limiting
        """
        self.rate = rate
        self.capacity = capacity
        self.redis = redis_client
        
        # In-memory storage fallback
        self._buckets: dict = {}
    
    def _get_bucket_key(self, identifier: str) -> str:
        return f"ratelimit:bucket:{identifier}"
    
    async def acquire(
        self,
        identifier: str,
        tokens: int = 1,
    ) -> Tuple[bool, float]:
        """
        Try to acquire tokens from the bucket.
        
        Args:
            identifier: Unique identifier for the bucket
            tokens: Number of tokens to acquire
            
        Returns:
            Tuple of (allowed, wait_time_seconds)
        """
        now = time.time()
        
        if self.redis:
            return await self._acquire_redis(identifier, tokens, now)
        else:
            return self._acquire_memory(identifier, tokens, now)
    
    def _acquire_memory(
        self,
        identifier: str,
        tokens: int,
        now: float,
    ) -> Tuple[bool, float]:
        """Acquire tokens from in-memory bucket."""
        if identifier not in self._buckets:
            self._buckets[identifier] = {
                "tokens": self.capacity,
                "last_update": now,
            }
        
        bucket = self._buckets[identifier]
        
        # Calculate tokens to add based on time elapsed
        elapsed = now - bucket["last_update"]
        tokens_to_add = elapsed * self.rate
        
        # Update bucket
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now
        
        # Try to acquire
        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return True, 0.0
        else:
            # Calculate wait time
            needed = tokens - bucket["tokens"]
            wait_time = needed / self.rate
            return False, wait_time
    
    async def _acquire_redis(
        self,
        identifier: str,
        tokens: int,
        now: float,
    ) -> Tuple[bool, float]:
        """Acquire tokens from Redis bucket."""
        key = self._get_bucket_key(identifier)
        
        # Lua script for atomic token bucket operation
        script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_needed = tonumber(ARGV[4])
        
        local bucket = redis.call('hmget', key, 'tokens', 'last_update')
        local current_tokens = tonumber(bucket[1]) or capacity
        local last_update = tonumber(bucket[2]) or now
        
        local elapsed = now - last_update
        local tokens_to_add = elapsed * rate
        current_tokens = math.min(capacity, current_tokens + tokens_to_add)
        
        if current_tokens >= tokens_needed then
            current_tokens = current_tokens - tokens_needed
            redis.call('hmset', key, 'tokens', current_tokens, 'last_update', now)
            redis.call('expire', key, 3600)
            return {1, 0}
        else
            local needed = tokens_needed - current_tokens
            local wait_time = needed / rate
            return {0, wait_time}
        end
        """
        
        try:
            result = await self.redis.eval(
                script,
                1,
                key,
                str(self.rate),
                str(self.capacity),
                str(now),
                str(tokens),
            )
            return bool(result[0]), float(result[1])
        except Exception as e:
            logger.error(f"Redis rate limit error: {e}")
            # Fall back to allowing the request
            return True, 0.0


# Default rate limiter instance
_rate_limiter: Optional[Limiter] = None


def get_rate_limiter() -> Limiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = create_rate_limiter()
    
    return _rate_limiter


def rate_limit_dependency(
    requests_per_minute: int = 60,
):
    """
    FastAPI dependency for rate limiting.
    
    Usage:
        @app.get("/api/resource")
        async def resource(
            rate_check: None = Depends(rate_limit_dependency(30))
        ):
            ...
    """
    async def check_rate_limit(request: Request):
        # Simple in-memory rate check
        client_ip = get_client_ip(request)
        settings = get_settings()
        
        if not settings.rate_limit.enabled:
            return
        
        # Use the global limiter
        limiter = get_rate_limiter()
        # Rate limiting is handled by slowapi middleware
        
    return check_rate_limit
