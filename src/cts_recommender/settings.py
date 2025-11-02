from pathlib import Path
from typing import Literal, Optional, List

from pydantic import AnyHttpUrl, BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class TMDBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TMDB_", extra="ignore")
    api_key: SecretStr
    api_base_url: AnyHttpUrl

class Settings(BaseSettings):

    # ---- Data roots (handy for pipelines/scripts) ----
    project_root: Path = Path(".").resolve()
    data_root: Path = Path("data")
    raw_dir: Path = data_root / "raw"
    interim_dir: Path = data_root / "interim"
    processed_dir: Path = data_root / "processed"
    reference_dir: Path = data_root / "reference"
    models_dir: Path = data_root / "models"

    # ----- Datasets -----
    original_rdata_programming: Path = raw_dir / "original_R_dataset.RData"

    # ---- app/runtime ----
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    verify_ssl: bool = False

    # ---- reproducibility ----
    random_seed: int = 42

    # ---- API server configuration ----
    api_host: str = "127.0.0.1"  # localhost for dev, 0.0.0.0 for docker/prod
    api_port: int = 8000
    api_reload: bool = True  # Auto-reload on code changes (dev only)
    api_workers: int = 1  # Number of uvicorn workers (increase for prod)
    cors_origins: List[str] = ["*"]  # Allowed CORS origins (restrict in prod)

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_prefix="APP_",      # APP_ENV, APP_LOG_LEVEL, etc.
        env_nested_delimiter='__',
        extra = "ignore"
    )

    # ---- integrations ----
    tmdb: Optional[TMDBSettings] = None  # <-- DO NOT instantiate here


def get_settings() -> Settings:
    """Singleton accessor to avoid reparsing .env on every import."""
    return Settings()