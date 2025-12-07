"""
Groq API client wrapper.

Provides async interface to Groq's ultra-fast LLM inference.
"""

import logging
import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from collections import defaultdict

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM completion."""
    
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    latency_ms: int


class GroqAPIError(Exception):
    """Base exception for Groq API errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class GroqRateLimitError(GroqAPIError):
    """Rate limit exceeded error."""
    pass


class GroqAuthenticationError(GroqAPIError):
    """Authentication error."""
    pass


class GroqClient:
    """
    Async client for Groq API with multi-key rotation.
    
    Provides high-performance LLM inference with automatic retries,
    error handling, and API key rotation for load distribution.
    """
    
    BASE_URL = "https://api.groq.com/openai/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        settings = get_settings()
        
        # Get all available API keys
        if api_keys:
            self.api_keys = api_keys
        else:
            self.api_keys = settings.groq.api_keys
        
        # Fallback to single key if provided
        if not self.api_keys and api_key:
            self.api_keys = [api_key]
        
        if not self.api_keys:
            raise ValueError("No Groq API keys provided. Set GROQ_API_KEY or GROQ_API_KEY1/2/3")
        
        self.model = model or settings.groq.model
        self.max_tokens = max_tokens or settings.groq.max_tokens
        self.temperature = temperature or settings.groq.temperature
        self.timeout = timeout or settings.groq.timeout
        
        # Key rotation state
        self._current_key_index = 0
        self._key_lock = asyncio.Lock()
        self._key_usage = defaultdict(int)  # Track usage per key
        self._key_errors = defaultdict(int)  # Track errors per key
        self._key_rate_limited_until = defaultdict(float)  # Track when keys can be retried
        self._key_clients: Dict[str, httpx.AsyncClient] = {}
        
        logger.info(f"Initialized GroqClient with {len(self.api_keys)} API key(s)")
    
    def _reset_key_errors_after_timeout(self):
        """Reset error counts for keys that haven't had errors recently."""
        current_time = time.time()
        # Reset errors for keys that haven't been rate limited in the last 5 minutes
        for key in list(self._key_errors.keys()):
            if (self._key_errors[key] > 0 and 
                self._key_rate_limited_until.get(key, 0) < current_time - 300):
                self._key_errors[key] = 0
                logger.debug(f"Reset error count for key {key[:10]}...")
    
    def _get_client_for_key(self, api_key: str) -> httpx.AsyncClient:
        """Get or create HTTP client for a specific API key."""
        if api_key not in self._key_clients or self._key_clients[api_key].is_closed:
            self._key_clients[api_key] = httpx.AsyncClient(
                base_url=self.BASE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout),
            )
        return self._key_clients[api_key]
    
    async def _get_next_key(self, agent_role: Optional[str] = None) -> str:
        """
        Get next API key using round-robin rotation.
        
        Args:
            agent_role: Optional agent role for consistent key assignment
            
        Returns:
            API key to use
        """
        async with self._key_lock:
            if len(self.api_keys) == 1:
                return self.api_keys[0]
            
            # If agent_role provided, use consistent hashing for same agent
            if agent_role:
                key_index = hash(agent_role) % len(self.api_keys)
                selected_key = self.api_keys[key_index]
            else:
                # Round-robin rotation
                selected_key = self.api_keys[self._current_key_index]
                self._current_key_index = (self._current_key_index + 1) % len(self.api_keys)
            
            self._key_usage[selected_key] += 1
            return selected_key
    
    async def _get_available_key(self, exclude_keys: Optional[List[str]] = None) -> Optional[str]:
        """
        Get an available API key, excluding rate-limited or failed keys.
        
        Args:
            exclude_keys: Keys to exclude from selection
            
        Returns:
            Available API key or None
        """
        exclude_keys = exclude_keys or []
        current_time = time.time()
        
        # Filter out keys that are:
        # 1. In exclude list
        # 2. Have too many errors
        # 3. Are still rate limited (waiting for retry-after)
        available_keys = [
            key for key in self.api_keys
            if (key not in exclude_keys 
                and self._key_errors[key] < 5
                and self._key_rate_limited_until.get(key, 0) <= current_time)
        ]
        
        if not available_keys:
            # Check if any keys are just temporarily rate limited
            rate_limited_keys = [
                key for key in self.api_keys
                if key not in exclude_keys and self._key_rate_limited_until.get(key, 0) > current_time
            ]
            
            if rate_limited_keys:
                # Find the key that will be available soonest
                soonest_available = min(
                    rate_limited_keys,
                    key=lambda k: self._key_rate_limited_until[k]
                )
                wait_time = self._key_rate_limited_until[soonest_available] - current_time
                logger.info(f"All keys rate limited. Earliest retry in {wait_time:.1f}s")
                return None
            
            # Reset error counts if all keys have errors
            if all(self._key_errors[key] >= 5 for key in self.api_keys):
                logger.warning("All API keys have errors, resetting error counts")
                self._key_errors.clear()
                available_keys = [k for k in self.api_keys if k not in exclude_keys]
        
        if not available_keys:
            return None
        
        # Return key with least usage
        return min(available_keys, key=lambda k: self._key_usage[k])
    
    async def close(self) -> None:
        """Close all HTTP clients."""
        for client in self._key_clients.values():
            if not client.is_closed:
                await client.aclose()
        self._key_clients.clear()
    
    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        agent_role: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM with automatic key rotation.
        
        Args:
            system_prompt: The system prompt setting agent behavior
            user_prompt: The user's prompt/query
            model: Override default model
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stop: Optional stop sequences
            agent_role: Optional agent role for consistent key assignment
            
        Returns:
            LLMResponse with completion content and metadata
        """
        import time
        
        start_time = time.monotonic()
        
        payload = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        if stop:
            payload["stop"] = stop
        
        # Reset old error counts periodically
        self._reset_key_errors_after_timeout()
        
        # Try with different keys if rate limited
        excluded_keys = []
        max_attempts = len(self.api_keys)
        
        for attempt in range(max_attempts):
            try:
                # Get API key (consistent for same agent, or round-robin)
                api_key = await self._get_next_key(agent_role=agent_role)
                
                # Skip if this key was already tried and failed
                if api_key in excluded_keys:
                    api_key = await self._get_available_key(exclude_keys=excluded_keys)
                    if not api_key:
                        raise GroqAPIError("All API keys exhausted", 429)
                
                client = self._get_client_for_key(api_key)
                response = await client.post("/chat/completions", json=payload)
                
                if response.status_code == 401:
                    self._key_errors[api_key] += 1
                    excluded_keys.append(api_key)
                    logger.warning(f"Authentication failed for key {api_key[:10]}..., trying next key")
                    if attempt < max_attempts - 1:
                        continue
                    raise GroqAuthenticationError("Invalid API key")
                    
                elif response.status_code == 429:
                    self._key_errors[api_key] += 1
                    excluded_keys.append(api_key)
                    
                    # Check for Retry-After header
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait_seconds = int(retry_after)
                            self._key_rate_limited_until[api_key] = time.time() + wait_seconds
                            logger.warning(
                                f"Rate limit exceeded for key {api_key[:10]}... "
                                f"(retry after {wait_seconds}s), trying next key"
                            )
                        except ValueError:
                            # If Retry-After is not a number, use default wait
                            self._key_rate_limited_until[api_key] = time.time() + 60
                            logger.warning(
                                f"Rate limit exceeded for key {api_key[:10]}... "
                                f"(retry after 60s), trying next key"
                            )
                    else:
                        # Default wait time if no Retry-After header
                        self._key_rate_limited_until[api_key] = time.time() + 60
                        logger.warning(
                            f"Rate limit exceeded for key {api_key[:10]}... "
                            f"(retry after 60s), trying next key"
                        )
                    
                    if attempt < max_attempts - 1:
                        continue
                    
                    # All keys rate limited - check if we should wait
                    available_key = await self._get_available_key(exclude_keys=excluded_keys)
                    if available_key:
                        # Found a key that's not rate limited, try it
                        continue
                    
                    # Calculate wait time for earliest available key
                    rate_limited_keys = [
                        k for k in self.api_keys
                        if self._key_rate_limited_until.get(k, 0) > time.time()
                    ]
                    if rate_limited_keys:
                        earliest_retry = min(
                            rate_limited_keys,
                            key=lambda k: self._key_rate_limited_until[k]
                        )
                        wait_time = self._key_rate_limited_until[earliest_retry] - time.time()
                        error_msg = (
                            f"Rate limit exceeded on all keys. "
                            f"Please wait {int(wait_time)} seconds before retrying. "
                            f"Consider adding more API keys or reducing request frequency."
                        )
                        raise GroqRateLimitError(error_msg)
                    
                    raise GroqRateLimitError(
                        "Rate limit exceeded on all keys. "
                        "Please wait a moment and try again, or add more API keys."
                    )
                    
                elif response.status_code >= 400:
                    error_body = response.json() if response.content else {}
                    error_msg = error_body.get("error", {}).get("message", response.text)
                    
                    # If it's a rate limit or auth error, try next key
                    if response.status_code in (401, 429) and attempt < max_attempts - 1:
                        self._key_errors[api_key] += 1
                        excluded_keys.append(api_key)
                        continue
                    
                    raise GroqAPIError(error_msg, response.status_code)
                
                # Success - reset error count for this key
                self._key_errors[api_key] = 0
                
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                
                latency_ms = int((time.monotonic() - start_time) * 1000)
                
                logger.debug(
                    f"API call successful with key {api_key[:10]}... "
                    f"(usage: {self._key_usage[api_key]}, errors: {self._key_errors[api_key]})"
                )
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    model=data["model"],
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "unknown"),
                    latency_ms=latency_ms,
                )
                
            except httpx.TimeoutException:
                if attempt < max_attempts - 1:
                    excluded_keys.append(api_key)
                    logger.warning(f"Timeout with key {api_key[:10]}..., trying next key")
                    continue
                logger.warning(f"Groq API timeout after {self.timeout}s")
                raise
            except httpx.NetworkError as e:
                if attempt < max_attempts - 1:
                    excluded_keys.append(api_key)
                    logger.warning(f"Network error with key {api_key[:10]}..., trying next key: {e}")
                    continue
                logger.error(f"Groq API network error: {e}")
                raise
        
        # If we get here, all keys failed
        raise GroqAPIError("All API keys failed", 500)
    
    async def complete_multi_turn(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        agent_role: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion with multi-turn conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override default model
            temperature: Override default temperature
            max_tokens: Override default max tokens
            agent_role: Optional agent role for consistent key assignment
            
        Returns:
            LLMResponse with completion content and metadata
        """
        import time
        
        start_time = time.monotonic()
        
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        
        # Use same key rotation logic as complete()
        excluded_keys = []
        max_attempts = len(self.api_keys)
        
        for attempt in range(max_attempts):
            try:
                api_key = await self._get_next_key(agent_role=agent_role)
                
                if api_key in excluded_keys:
                    api_key = await self._get_available_key(exclude_keys=excluded_keys)
                    if not api_key:
                        raise GroqAPIError("All API keys exhausted", 429)
                
                client = self._get_client_for_key(api_key)
                response = await client.post("/chat/completions", json=payload)
                
                if response.status_code >= 400:
                    if response.status_code in (401, 429) and attempt < max_attempts - 1:
                        self._key_errors[api_key] += 1
                        excluded_keys.append(api_key)
                        continue
                    self._handle_error(response)
                
                self._key_errors[api_key] = 0
                
                data = response.json()
                choice = data["choices"][0]
                usage = data.get("usage", {})
                
                latency_ms = int((time.monotonic() - start_time) * 1000)
                
                return LLMResponse(
                    content=choice["message"]["content"],
                    model=data["model"],
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    finish_reason=choice.get("finish_reason", "unknown"),
                    latency_ms=latency_ms,
                )
                
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt < max_attempts - 1:
                    excluded_keys.append(api_key)
                    continue
                raise
        
        # If we get here, all keys failed
        raise GroqAPIError("All API keys failed", 500)
    
    def _handle_error(self, response: httpx.Response) -> None:
        """Handle API error responses."""
        if response.status_code == 401:
            raise GroqAuthenticationError("Invalid API key")
        elif response.status_code == 429:
            raise GroqRateLimitError("Rate limit exceeded")
        else:
            error_body = response.json() if response.content else {}
            error_msg = error_body.get("error", {}).get("message", response.text)
            raise GroqAPIError(error_msg, response.status_code)
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        try:
            # Use first available key for model listing
            api_key = await self._get_available_key()
            if not api_key:
                api_key = self.api_keys[0]
            
            client = self._get_client_for_key(api_key)
            response = await client.get("/models")
            
            if response.status_code >= 400:
                self._handle_error(response)
            
            data = response.json()
            return data.get("data", [])
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API connectivity and health for all keys."""
        import time
        
        health_results = []
        
        for api_key in self.api_keys:
            try:
                start = time.monotonic()
                client = self._get_client_for_key(api_key)
                response = await client.get("/models")
                latency_ms = int((time.monotonic() - start) * 1000)
                
                if response.status_code == 200:
                    data = response.json()
                    health_results.append({
                        "key": api_key[:10] + "...",
                        "status": "healthy",
                        "latency_ms": latency_ms,
                        "models_available": len(data.get("data", [])),
                    })
                else:
                    health_results.append({
                        "key": api_key[:10] + "...",
                        "status": "unhealthy",
                        "error": f"HTTP {response.status_code}",
                    })
            except Exception as e:
                health_results.append({
                    "key": api_key[:10] + "...",
                    "status": "unhealthy",
                    "error": str(e),
                })
        
        # Overall status
        healthy_count = sum(1 for r in health_results if r["status"] == "healthy")
        overall_status = "healthy" if healthy_count > 0 else "unhealthy"
        
        return {
            "status": overall_status,
            "keys_total": len(self.api_keys),
            "keys_healthy": healthy_count,
            "key_details": health_results,
        }


# Singleton client instance
_groq_client: Optional[GroqClient] = None


def get_groq_client() -> GroqClient:
    """Get or create the Groq client singleton."""
    global _groq_client
    
    if _groq_client is None:
        _groq_client = GroqClient()
    
    return _groq_client


async def close_groq_client() -> None:
    """Close the Groq client."""
    global _groq_client
    
    if _groq_client is not None:
        await _groq_client.close()
        _groq_client = None
