from __future__ import annotations

import argparse
from dataclasses import replace

from .config import Settings, save_provider_settings_to_dotenv
from .pipeline import run_pipeline


def _selected_provider(settings: Settings, override: str | None) -> str:
    return "gemini"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a DDR report from inspection and thermal PDFs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate DDR output files.")
    generate.add_argument("--inspection", required=True, help="Path to the inspection PDF.")
    generate.add_argument("--thermal", required=True, help="Path to the thermal PDF.")
    generate.add_argument("--out-dir", required=True, help="Directory where run artifacts should be written.")
    generate.add_argument("--model", help="Override the Gemini model name.")
    generate.add_argument("--enable-ocr", action="store_true", help="Enable OCR fallback for text-poor pages.")
    generate.add_argument("--no-llm", action="store_true", help="Disable LLM usage and use deterministic fallback only.")
    generate.add_argument("--api-key", help="Override the Gemini API key for this run.")
    generate.add_argument(
        "--persist-api-key",
        action="store_true",
        help="Persist the supplied --api-key into the local .env file.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()
    provider = _selected_provider(settings, None)
    settings = replace(settings, llm_provider=provider)
    if args.model:
        settings = replace(settings, gemini_model=args.model)
    if args.api_key:
        settings = replace(settings, gemini_api_key=args.api_key)
        if args.persist_api_key:
            save_provider_settings_to_dotenv(provider, args.api_key, model=settings.selected_model)
    if args.enable_ocr:
        settings = replace(settings, enable_ocr=True)
    settings = replace(settings, llm_enabled=bool(settings.selected_api_key))
    if args.no_llm:
        settings = replace(settings, llm_enabled=False)

    result = run_pipeline(
        inspection_path=args.inspection,
        thermal_path=args.thermal,
        out_dir=args.out_dir,
        settings=settings,
    )
    print(f"Run directory: {result.run_dir}")
    print(f"Report JSON: {result.report_json_path}")
    print(f"Report HTML: {result.report_html_path}")
    print(f"Report PDF: {result.report_pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
