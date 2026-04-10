"""Configuration management for the ASR backend."""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "info"

    # Model settings - OpenVINO
    model_path: str = "./converted_models/Qwen3-ASR-1.7B-OV"
    ov_device: str = "CPU"  # OpenVINO device: "CPU", "GPU", "NPU"
    ov_max_batch_size: int = 1
    ov_max_new_tokens: int = 512

    # Legacy model settings (kept for compatibility, not used with OpenVINO)
    n_ctx: int = 8192
    n_threads: int = 0  # 0 = auto-detect
    n_batch: int = 1
    quantization: str = "8bit"  # Options: "none", "4bit", "8bit"
    use_gguf: bool = False      # Use GGUF format via llama.cpp (best for CPU)

    # Session settings
    max_concurrent_sessions: int = 4
    max_buffer_ms: int = 5000

    # WebSocket settings
    ws_ping_interval: int = 20
    ws_ping_timeout: int = 10

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()

    # Auto-detect threads if not set
    if settings.n_threads == 0:
        import os
        settings.n_threads = os.cpu_count() or 4

    return settings