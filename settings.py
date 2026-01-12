# settings.py
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _get_env(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else v.strip()


def _get_int(name: str, default: int, *, min_value: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        v = default
    else:
        try:
            v = int(raw.strip())
        except ValueError as e:
            raise ValueError(f"Env var {name} must be an int, got {raw!r}") from e

    if min_value is not None and v < min_value:
        raise ValueError(f"Env var {name} must be >= {min_value}, got {v}")
    return v


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Env var {name} must be a boolean-like value "
        f"(true/false/1/0/yes/no/on/off), got {raw!r}"
    )


def _load_dotenv_if_enabled() -> None:
    """
    Production-safe dotenv loading:
    - Default: OFF
    - Enable explicitly with LOAD_DOTENV=true (for local dev)
    - Doesn't override already-set env vars.
    """
    if _get_bool("LOAD_DOTENV", False):
        load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    # Core
    env: str = field(default_factory=lambda: _get_env("ENV", "dev"))
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO"))

    # Data
    pdf_dir: Path = field(default_factory=lambda: Path(_get_env("PDF_DIR", "data/raw/pdfs")))
    artifacts_dir: Path = field(default_factory=lambda: Path(_get_env("ARTIFACTS_DIR", "data/staging")))
    manifest_dir: Path = field(default_factory=lambda: Path(_get_env("MANIFEST_DIR", "data/processed/manifest")))

    # Chunking
    chunk_chars: int = field(default_factory=lambda: _get_int("CHUNK_CHARS", 1200, min_value=1))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 200, min_value=0))

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: _get_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    embed_batch_size: int = field(default_factory=lambda: _get_int("EMBED_BATCH_SIZE", 32, min_value=1))
    normalize_embeddings: bool = field(default_factory=lambda: _get_bool("NORMALIZE_EMBEDDINGS", True))

    # Qdrant
    qdrant_host: str = field(default_factory=lambda: _get_env("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: _get_int("QDRANT_PORT", 6333, min_value=1))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    qdrant_collection: str = field(default_factory=lambda: _get_env("QDRANT_COLLECTION", "rag_collection"))
    qdrant_distance: str = field(default_factory=lambda: _get_env("QDRANT_DISTANCE", "cosine").lower())
    upsert_batch_size: int = field(default_factory=lambda: _get_int("UPSERT_BATCH_SIZE", 64, min_value=1))

    def validate(self) -> "Settings":
        allowed_envs = {"dev", "staging", "prod", "test"}
        if self.env.lower() not in allowed_envs:
            raise ValueError(f"ENV must be one of {sorted(allowed_envs)}, got {self.env!r}")

        allowed_distances = {"cosine", "dot", "euclid"}
        if self.qdrant_distance not in allowed_distances:
            raise ValueError(
                f"QDRANT_DISTANCE must be one of {sorted(allowed_distances)}, got {self.qdrant_distance!r}"
            )

        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError(
                f"CHUNK_OVERLAP must be < CHUNK_CHARS, got overlap={self.chunk_overlap}, chars={self.chunk_chars}"
            )

        # Optional: ensure paths are sensible (don’t force-exist in prod, but normalize)
        # You can enforce existence in dev if you want:
        # if self.env == "dev" and not self.pdf_dir.exists():
        #     raise ValueError(f"PDF_DIR does not exist: {self.pdf_dir}")

        return self


# ---- module entrypoint ----
# Load dotenv only if explicitly enabled (safe for prod)
_load_dotenv_if_enabled()

# Create a single settings instance for the app to import
settings = Settings().validate()
from settings import settings

print(settings.qdrant_host)
