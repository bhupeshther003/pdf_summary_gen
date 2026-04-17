from __future__ import annotations

from pathlib import Path

from ddr_report.config import save_gemini_key_to_dotenv, save_provider_settings_to_dotenv


def test_save_gemini_key_to_dotenv_updates_or_creates_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"

    first_path = save_gemini_key_to_dotenv("gm-test-1", model="gemini-2.5-flash", env_path=env_path)
    assert first_path == env_path
    text = env_path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=gm-test-1" in text
    assert "GEMINI_MODEL=gemini-2.5-flash" in text
    assert "LLM_PROVIDER=gemini" in text

    save_gemini_key_to_dotenv("gm-test-2", model="gemini-2.5-flash", env_path=env_path)
    text = env_path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=gm-test-2" in text
    assert text.count("GEMINI_API_KEY=") == 1


def test_save_provider_settings_to_dotenv_writes_gemini(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"

    save_provider_settings_to_dotenv("gemini", "gm-test-1", model="gemini-2.5-flash", env_path=env_path)

    text = env_path.read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=gm-test-1" in text
    assert "GEMINI_MODEL=gemini-2.5-flash" in text
    assert "LLM_PROVIDER=gemini" in text
