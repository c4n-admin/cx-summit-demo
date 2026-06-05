import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    api_base: str = os.getenv("LLM_API_BASE", "http://localhost:8001").rstrip("/")
    api_key: str = os.getenv("LLM_API_KEY", "lm-studio")
    model: str = os.getenv("LLM_MODEL", "local-model")
    timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "600"))


@dataclass(frozen=True)
class AppConfig:
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("APP_PORT", "8080"))