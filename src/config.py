"""Configuração central da aplicação; segredos são obtidos apenas do ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    """Erro de configuração sem expor valores de variáveis de ambiente."""


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str = field(repr=False)
    model_fast: str
    model_strong: str
    model_vision: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int


@dataclass(frozen=True, slots=True)
class IngestionSettings:
    max_document_mb: int
    ocr_languages: str
    min_native_chars: int
    ocr_timeout_seconds: float
    render_dpi: int
    processed_dir: Path


@dataclass(frozen=True, slots=True)
class StorageSettings:
    sqlite_path: Path
    chroma_path: Path
    chroma_collection: str
    embedding_model: str
    embedding_device: str
    chunk_size_chars: int
    chunk_overlap_chars: int
    rag_top_k: int


def get_settings(*, require_api_key: bool = True) -> Settings:
    """Carrega as configurações do .env e do ambiente, sem sobrescrever o ambiente."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise ConfigurationError("Defina GROQ_API_KEY no arquivo .env.")

    def required(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            raise ConfigurationError(
                f"Defina {name} no arquivo .env (copie .env.example para .env)."
            )
        return value

    try:
        temperature = float(os.getenv("GROQ_TEMPERATURE", "0"))
        max_tokens = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
        timeout_seconds = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))
        max_retries = int(os.getenv("GROQ_MAX_RETRIES", "3"))
    except ValueError as error:
        raise ConfigurationError(
            "GROQ_TEMPERATURE, GROQ_MAX_TOKENS, GROQ_TIMEOUT_SECONDS e "
            "GROQ_MAX_RETRIES devem conter valores numéricos válidos."
        ) from error

    if not 0 <= temperature <= 2:
        raise ConfigurationError("GROQ_TEMPERATURE deve estar entre 0 e 2.")
    if max_tokens < 1 or timeout_seconds <= 0 or not 1 <= max_retries <= 3:
        raise ConfigurationError(
            "GROQ_MAX_TOKENS e GROQ_TIMEOUT_SECONDS devem ser positivos; "
            "GROQ_MAX_RETRIES deve estar entre 1 e 3."
        )

    return Settings(
        groq_api_key=api_key,
        model_fast=required("GROQ_MODEL_FAST"),
        model_strong=required("GROQ_MODEL_STRONG"),
        model_vision=required("GROQ_MODEL_VISION"),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def get_ingestion_settings() -> IngestionSettings:
    """Carrega parâmetros locais de ingestão sem exigir configuração/chave Groq."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    try:
        settings = IngestionSettings(
            max_document_mb=int(os.getenv("MAX_DOCUMENT_MB", "30")),
            ocr_languages=os.getenv("OCR_LANGUAGES", "por+eng").strip(),
            min_native_chars=int(os.getenv("OCR_MIN_NATIVE_CHARS", "20")),
            ocr_timeout_seconds=float(os.getenv("OCR_TIMEOUT_SECONDS", "60")),
            render_dpi=int(os.getenv("OCR_RENDER_DPI", "200")),
            processed_dir=PROJECT_ROOT / "data" / "processed",
        )
    except ValueError as error:
        raise ConfigurationError(
            "MAX_DOCUMENT_MB, OCR_MIN_NATIVE_CHARS, OCR_TIMEOUT_SECONDS e "
            "OCR_RENDER_DPI devem conter valores numéricos válidos."
        ) from error

    if (
        settings.max_document_mb < 1
        or settings.min_native_chars < 1
        or settings.ocr_timeout_seconds <= 0
        or settings.render_dpi < 72
        or not settings.ocr_languages
    ):
        raise ConfigurationError("As configurações de ingestão/OCR devem ser positivas.")
    return settings


def get_storage_settings() -> StorageSettings:
    """Carrega opções de persistência local sem chave ou conexão externa."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    try:
        settings = StorageSettings(
            sqlite_path=PROJECT_ROOT / "data" / "processed" / "policies.sqlite3",
            chroma_path=PROJECT_ROOT / "data" / "processed" / "chroma",
            chroma_collection=os.getenv(
                "CHROMA_COLLECTION_NAME", "do_policy_chunks_v1"
            ).strip(),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ).strip(),
            embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu").strip(),
            chunk_size_chars=int(os.getenv("VECTOR_CHUNK_SIZE_CHARS", "1200")),
            chunk_overlap_chars=int(os.getenv("VECTOR_CHUNK_OVERLAP_CHARS", "150")),
            rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        )
    except ValueError as error:
        raise ConfigurationError(
            "VECTOR_CHUNK_SIZE_CHARS, VECTOR_CHUNK_OVERLAP_CHARS e RAG_TOP_K "
            "devem conter valores inteiros válidos."
        ) from error
    if (
        not settings.chroma_collection
        or not settings.embedding_model
        or settings.embedding_device not in {"cpu", "cuda", "mps"}
        or settings.chunk_size_chars < 200
        or settings.chunk_overlap_chars < 0
        or settings.chunk_overlap_chars >= settings.chunk_size_chars
        or settings.rag_top_k < 1
    ):
        raise ConfigurationError("Configurações locais de armazenamento/RAG inválidas.")
    return settings
