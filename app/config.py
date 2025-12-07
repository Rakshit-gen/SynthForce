"""
Configuration management for Synthetic Workforce Simulator.

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    model_config = SettingsConfigDict(env_prefix="DB_")
    
    url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/workforce_simulator",
        alias="DATABASE_URL"
    )
    pool_min: int = Field(default=5, ge=1, le=50)
    pool_max: int = Field(default=20, ge=5, le=100)
    pool_timeout: int = Field(default=30, ge=5, le=120)
    echo: bool = Field(default=False)


class RedisSettings(BaseSettings):
    """Redis configuration."""
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")
    
    url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    max_connections: int = Field(default=10, ge=1, le=100)
    decode_responses: bool = Field(default=True)


class GroqSettings(BaseSettings):
    """Groq API configuration."""
    
    model_config = SettingsConfigDict(env_prefix="GROQ_")
    
    api_key: str = Field(default="", alias="GROQ_API_KEY")
    api_key_1: Optional[str] = Field(default=None, alias="GROQ_API_KEY1")
    api_key_2: Optional[str] = Field(default=None, alias="GROQ_API_KEY2")
    api_key_3: Optional[str] = Field(default=None, alias="GROQ_API_KEY3")
    model: str = Field(default="llama-3.3-70b-versatile")
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=1, le=10)
    
    @property
    def api_keys(self) -> List[str]:
        """Get all available API keys."""
        keys = []
        if self.api_key:
            keys.append(self.api_key)
        if self.api_key_1:
            keys.append(self.api_key_1)
        if self.api_key_2:
            keys.append(self.api_key_2)
        if self.api_key_3:
            keys.append(self.api_key_3)
        return keys if keys else [self.api_key] if self.api_key else []
    
    @property
    def available_models(self) -> List[str]:
        """List of available Groq models."""
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""
    
    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_")
    
    requests: int = Field(default=100, ge=1, le=10000)
    window: int = Field(default=60, ge=1, le=3600)
    enabled: bool = Field(default=True)


class SimulationSettings(BaseSettings):
    """Simulation engine configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    max_turns: int = Field(default=50, ge=1, le=500, alias="MAX_SIMULATION_TURNS")
    max_concurrent: int = Field(default=10, ge=1, le=100, alias="MAX_CONCURRENT_SIMULATIONS")
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440, alias="SESSION_TIMEOUT_MINUTES")
    memory_retention_days: int = Field(default=30, ge=1, le=365, alias="MEMORY_RETENTION_DAYS")


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    level: str = Field(default="INFO", alias="LOG_LEVEL")
    json_format: bool = Field(default=True, alias="JSON_LOGS")
    
    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return upper_v


class SecuritySettings(BaseSettings):
    """Security configuration."""
    
    model_config = SettingsConfigDict(env_prefix="")
    
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        alias="CORS_ORIGINS"
    )
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        alias="SECRET_KEY"
    )
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


class Settings(BaseSettings):
    """Main application settings aggregating all configuration."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    app_name: str = Field(default="Synthetic Workforce Simulator", alias="APP_NAME")
    api_version: str = Field(default="v1", alias="API_VERSION")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")
    
    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    workers: int = Field(default=4, alias="WORKERS")
    
    # Metrics
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")
    
    # Sub-configurations (loaded separately for better organization)
    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings()
    
    @property
    def redis(self) -> RedisSettings:
        return RedisSettings()
    
    @property
    def groq(self) -> GroqSettings:
        return GroqSettings()
    
    @property
    def rate_limit(self) -> RateLimitSettings:
        return RateLimitSettings()
    
    @property
    def simulation(self) -> SimulationSettings:
        return SimulationSettings()
    
    @property
    def logging(self) -> LoggingSettings:
        return LoggingSettings()
    
    @property
    def security(self) -> SecuritySettings:
        return SecuritySettings()
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    """
    return Settings()


# Convenience function for dependency injection
def get_config() -> Settings:
    """Get settings for FastAPI dependency injection."""
    return get_settings()
