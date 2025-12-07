"""
Security utilities and middleware.

Provides CORS configuration and security headers.
"""

import logging
from typing import Callable, List

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings

logger = logging.getLogger(__name__)


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the application.
    
    Allows cross-origin requests from configured origins.
    """
    settings = get_settings()
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=[
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-Correlation-ID",
        ],
        expose_headers=[
            "X-Request-ID",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ],
        max_age=600,  # Cache preflight for 10 minutes
    )
    
    logger.info(
        f"CORS configured for origins: {settings.security.cors_origins_list}"
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "accelerometer=(), camera=(), geolocation=(), "
            "gyroscope=(), magnetometer=(), microphone=(), "
            "payment=(), usb=()"
        )
        
        # Content Security Policy (adjust based on your frontend needs)
        if not request.url.path.startswith("/docs"):
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self' https:;"
            )
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add request ID to all requests.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        import uuid
        
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        
        # Add to request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track request timing.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        import time
        
        start_time = time.monotonic()
        
        response = await call_next(request)
        
        duration_ms = (time.monotonic() - start_time) * 1000
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        # Log slow requests
        if duration_ms > 1000:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration_ms:.2f}ms"
            )
        
        return response


def configure_security(app: FastAPI) -> None:
    """
    Configure all security middleware.
    
    Should be called during app initialization.
    """
    # Add middleware in order (last added = first executed)
    app.add_middleware(RequestTimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Configure CORS
    configure_cors(app)
    
    logger.info("Security middleware configured")


def validate_api_key(api_key: str) -> bool:
    """
    Validate an API key.
    
    This is a placeholder for actual API key validation.
    In production, this would check against a database or secret store.
    """
    settings = get_settings()
    
    # For now, just check it's not empty
    # In production, implement proper validation
    return bool(api_key and len(api_key) >= 32)


def sanitize_input(value: str, max_length: int = 10000) -> str:
    """
    Sanitize user input.
    
    Removes potentially dangerous content and enforces length limits.
    """
    if not value:
        return ""
    
    # Truncate to max length
    value = value[:max_length]
    
    # Remove null bytes
    value = value.replace("\x00", "")
    
    # Normalize whitespace
    value = " ".join(value.split())
    
    return value
