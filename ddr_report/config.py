from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


SUPPORTED_PROVIDERS = ("gemini",)
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
}
PROVIDER_KEY_ENV = {"gemini": "GEMINI_API_KEY"}
PROVIDER_MODEL_ENV = {"gemini": "GEMINI_MODEL"}


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _streamlit_secret(name: str) -> str | None:
    try:
        import streamlit as st
    except Exception:
        return None
    try:
        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value else None
    except Exception:
        return None
    return None


def load_runtime_env() -> None:
    load_dotenv(_project_root() / ".env", override=False)
    load_dotenv(override=False)


def _env_or_secret(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
        value = _streamlit_secret(name)
        if value:
            return value
    return None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalize_provider(raw: str | None) -> str | None:
    if not raw:
        return None
    provider = raw.strip().lower()
    if provider in SUPPORTED_PROVIDERS:
        return provider
    if provider in {"google", "google-genai"}:
        return "gemini"
    return None


def _select_provider(preferred_provider: str | None, gemini_api_key: str | None) -> str:
    provider = normalize_provider(preferred_provider)
    if provider:
        return provider
    if gemini_api_key:
        return "gemini"
    return "none"


@dataclass(slots=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    llm_provider: str
    enable_ocr: bool
    debug_save_intermediate: bool
    llm_enabled: bool

    @property
    def selected_api_key(self) -> str | None:
        return self.gemini_api_key if self.llm_provider == "gemini" else None

    @property
    def selected_model(self) -> str:
        return self.gemini_model

    @classmethod
    def from_env(cls) -> "Settings":
        load_runtime_env()
        gemini_api_key = _env_or_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
        preferred_provider = _env_or_secret("LLM_PROVIDER")
        llm_provider = _select_provider(preferred_provider, gemini_api_key)
        gemini_model = _env_or_secret("GEMINI_MODEL") or DEFAULT_MODELS["gemini"]
        return cls(
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            llm_provider=llm_provider,
            enable_ocr=_env_flag("ENABLE_OCR", default=False),
            debug_save_intermediate=_env_flag("DEBUG_SAVE_INTERMEDIATE", default=True),
            llm_enabled=bool(llm_provider == "gemini" and gemini_api_key),
        )


def save_provider_settings_to_dotenv(
    provider: str,
    api_key: str,
    model: str | None = None,
    env_path: Path | None = None,
) -> Path:
    normalized_provider = normalize_provider(provider)
    if normalized_provider != "gemini":
        raise ValueError(f"Unsupported provider: {provider}")

    env_path = env_path or (_project_root() / ".env")
    lines: list[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    updates = {
        PROVIDER_KEY_ENV[normalized_provider]: api_key,
        PROVIDER_MODEL_ENV[normalized_provider]: model or DEFAULT_MODELS[normalized_provider],
        "LLM_PROVIDER": normalized_provider,
    }
    seen = set()
    output: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip()
        if key in updates:
            output.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in updates.items():
        if key not in seen:
            output.append(f"{key}={value}")

    env_path.write_text("\n".join(output).strip() + "\n", encoding="utf-8")
    for key, value in updates.items():
        os.environ[key] = value
    return env_path


def save_gemini_key_to_dotenv(api_key: str, model: str | None = None, env_path: Path | None = None) -> Path:
    return save_provider_settings_to_dotenv("gemini", api_key=api_key, model=model, env_path=env_path)
