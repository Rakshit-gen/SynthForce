"""
Main application entry point.

Creates and configures the FastAPI application.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.core.simulation import router as simulation_router
from app.db import init_database, close_database, check_database_health
from app.services import close_groq_client, get_groq_client
from app.utils import configure_logging, configure_security, get_rate_limiter

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"]
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan handler.
    
    Handles startup and shutdown events.
    """
    settings = get_settings()
    
    # Startup
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    
    # Initialize database
    await init_database()
    logger.info("Database initialized")
    
    # Warm up LLM client
    groq_client = get_groq_client()
    health = await groq_client.health_check()
    if health["status"] == "healthy":
        logger.info(f"Groq API connected, {health.get('models_available', 0)} models available")
    else:
        logger.warning(f"Groq API health check failed: {health.get('error')}")
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    await close_groq_client()
    await close_database()
    
    logger.info("Application shutdown complete")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    settings = get_settings()
    
    # Configure logging
    configure_logging()
    
    # Create application
    app = FastAPI(
        title=settings.app_name,
        description="Multi-agent simulation engine for workforce scenarios",
        version=settings.api_version,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        openapi_url="/openapi.json" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # Configure security (CORS, headers)
    configure_security(app)
    
    # Add rate limiter
    limiter = get_rate_limiter()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Register routes
    register_routes(app)
    
    # Add metrics middleware
    if settings.metrics_enabled:
        add_metrics_middleware(app)
    
    return app


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers."""
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        """Handle validation errors."""
        errors = []
        for error in exc.errors():
            errors.append({
                "field": ".".join(str(loc) for loc in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "validation_error",
                "message": "Request validation failed",
                "details": errors,
            },
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        """Handle uncaught exceptions."""
        logger.exception(f"Unhandled exception: {exc}")
        
        settings = get_settings()
        
        content = {
            "error": "internal_error",
            "message": "An unexpected error occurred",
        }
        
        if settings.debug:
            content["detail"] = str(exc)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=content,
        )


def register_routes(app: FastAPI) -> None:
    """Register API routes."""
    
    # Health check endpoint
    @app.get("/health", tags=["system"])
    async def health_check():
        """
        Health check endpoint.
        
        Returns the status of the application and its dependencies.
        """
        settings = get_settings()
        
        # Check database
        db_health = await check_database_health()
        
        # Check Groq API
        groq_client = get_groq_client()
        groq_health = await groq_client.health_check()
        
        # Determine overall status
        components_healthy = (
            db_health["status"] == "healthy" and
            groq_health["status"] == "healthy"
        )
        
        return {
            "status": "healthy" if components_healthy else "degraded",
            "version": settings.api_version,
            "environment": settings.environment,
            "components": [
                {
                    "name": "database",
                    **db_health,
                },
                {
                    "name": "groq_api",
                    **groq_health,
                },
            ],
        }
    
    # Ready check endpoint
    @app.get("/ready", tags=["system"])
    async def ready_check():
        """
        Readiness check for load balancers.
        """
        db_health = await check_database_health()
        
        if db_health["status"] != "healthy":
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"status": "not_ready", "reason": "database unavailable"},
            )
        
        return {"status": "ready"}
    
    # Metrics endpoint
    @app.get("/metrics", tags=["system"])
    async def metrics():
        """
        Prometheus metrics endpoint.
        """
        from fastapi.responses import Response
        
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )
    
    # Root endpoint
    @app.get("/", tags=["system"])
    async def root():
        """
        Root endpoint with API information.
        """
        settings = get_settings()
        
        return {
            "name": settings.app_name,
            "version": settings.api_version,
            "documentation": "/docs" if settings.debug else "disabled",
            "endpoints": {
                "health": "/health",
                "ready": "/ready",
                "simulate": {
                    "start": "POST /simulate/start",
                    "next": "POST /simulate/next",
                    "what_if": "POST /simulate/what-if",
                },
                "agents": "GET /agents/list",
                "memory": "GET /memory/{session_id}",
            },
        }
    
    # Register simulation routes
    app.include_router(simulation_router, prefix="")


def add_metrics_middleware(app: FastAPI) -> None:
    """Add Prometheus metrics middleware."""
    import time
    
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        # Record metrics
        duration = time.time() - start_time
        endpoint = request.url.path
        
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=endpoint,
            status=response.status_code,
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=endpoint,
        ).observe(duration)
        
        return response


# Create application instance
app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
