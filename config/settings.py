"""
config/settings.py
Central configuration using Pydantic Settings.
Reads from environment variables / .env file.

All configuration in one place — no scattered os.getenv() calls.
"""
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    from pydantic import BaseSettings, Field

# Project root — two levels up from this file
ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """
    All ARIA configuration. Values read from .env file or environment.
    Pydantic validates types and raises clear errors on misconfiguration.
    """

    # ── Project ───────────────────────────────────────────────────────
    project_name: str = "ARIA"
    version: str = "1.0.0"
    root_dir: Path = ROOT
    demo_mode: bool = Field(default=True, description="Use simulator instead of live APIs")

    # ── LLM ───────────────────────────────────────────────────────────
    anthropic_api_key: Optional[str] = Field(default=None)
    openai_api_key: Optional[str] = Field(default=None)

    # Which provider to use: "anthropic" | "openai"
    llm_provider: str = Field(default="openai")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.0)
    llm_max_tokens: int = Field(default=1024)

    # ── Data APIs ─────────────────────────────────────────────────────
    serpapi_key: Optional[str] = Field(default=None)
    trends_region: str = Field(default="US")
    trends_timeframe: str = Field(default="today 12-m")

    # ── Database ──────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite:///./aria.db")

    # ── Redis ─────────────────────────────────────────────────────────
    redis_url: Optional[str] = Field(default=None)
    # Cache TTL in seconds (default: 6 hours)
    cache_ttl_seconds: int = Field(default=86400)  # 24h — protects SerpAPI credits

    # ── Agent ─────────────────────────────────────────────────────────
    agent_schedule_minutes: int = Field(default=60)
    # Max % change agent can auto-execute without human approval
    agent_auto_approve_max_pct: float = Field(default=10.0)
    # Changes above this % go to human review queue
    agent_human_review_threshold_pct: float = Field(default=10.0)
    # Max LLM iterations per agent run
    agent_max_iterations: int = Field(default=6)

    # ── Monitoring ────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── AWS S3 Model Store ────────────────────────────────────────────
    aws_access_key_id:     Optional[str] = Field(default=None)
    aws_secret_access_key: Optional[str] = Field(default=None)
    s3_bucket_name:        Optional[str] = Field(default=None)
    s3_region:             str           = Field(default="us-east-1")
    s3_model_prefix:       str           = Field(default="aria-models")

    @property
    def has_s3(self) -> bool:
        return bool(
            self.aws_access_key_id and
            self.aws_secret_access_key and
            self.s3_bucket_name and
            "your" not in (self.aws_access_key_id or "").lower()
        )
    langfuse_enabled: bool = Field(default=False)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)

    # ── Paths ─────────────────────────────────────────────────────────
    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def processed_dir(self) -> Path:
        return self.root_dir / "data" / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.root_dir / "data" / "cache"

    @property
    def models_dir(self) -> Path:
        return self.root_dir / "models" / "saved_models"

    @property
    def prophet_dir(self) -> Path:
        return self.root_dir / "models" / "prophet" / "saved_models"

    @property
    def results_dir(self) -> Path:
        return self.root_dir / "results"

    @property
    def features_path(self) -> Path:
        return self.processed_dir / "features.parquet"

    @property
    def xgb_model_path(self) -> Path:
        return self.models_dir / "pricing_model.ubj"

    @property
    def xgb_meta_path(self) -> Path:
        return self.models_dir / "pricing_model_meta.json"

    def ensure_dirs(self):
        """Create all required directories if they don't exist."""
        dirs = [
            self.data_dir, self.processed_dir, self.cache_dir,
            self.models_dir, self.prophet_dir, self.results_dir,
            self.root_dir / "data" / "raw",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def llm_api_key(self) -> Optional[str]:
        """Returns the active LLM API key based on provider setting."""
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
        return self.openai_api_key

    @property
    def has_serpapi(self) -> bool:
        return bool(self.serpapi_key and self.serpapi_key != "your_serpapi_key_here")

    @property
    def has_redis(self) -> bool:
        return bool(self.redis_url)

    class Config:
        env_file = str(ROOT / ".env")
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"   # ignore unknown env vars like POSTGRES_USER etc
        env_nested_delimiter = "__"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        **kwargs,
    ):
        # Priority order (highest to lowest):
        # 1. init_settings — passed directly to constructor
        # 2. env_settings  — shell environment variables (export KEY=val)
        # 3. dotenv_settings — .env file
        # env_settings before dotenv_settings means shell export wins
        return (
            init_settings,
            env_settings,
            dotenv_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns cached settings instance.
    Use this everywhere instead of instantiating Settings() directly.

    Example:
        from config.settings import get_settings
        settings = get_settings()
        print(settings.demo_mode)
    """
    s = Settings()
    s.ensure_dirs()
    return s


# ── Quick validation on import ────────────────────────────────────────
if __name__ == "__main__":
    settings = get_settings()
    print(f"\n{'='*50}")
    print(f"ARIA Configuration")
    print(f"{'='*50}")
    print(f"  Demo mode       : {settings.demo_mode}")
    print(f"  LLM provider    : {settings.llm_provider}")
    print(f"  LLM model       : {settings.llm_model}")
    print(f"  Has SerpAPI     : {settings.has_serpapi}")
    print(f"  Has Redis       : {settings.has_redis}")
    print(f"  Database        : {settings.database_url}")
    print(f"  Root dir        : {settings.root_dir}")
    print(f"  Log level       : {settings.log_level}")
    print(f"  Agent schedule  : every {settings.agent_schedule_minutes} min")
    print(f"  Auto-approve    : changes ≤ {settings.agent_auto_approve_max_pct}%")
    print(f"{'='*50}\n")
