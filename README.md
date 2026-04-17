# DDR Report Generator

Python 3.11 pipeline for converting an inspection PDF and a thermal PDF into a client-ready DDR report in both JSON and PDF formats.

It now includes:

- a CLI workflow
- a Streamlit upload UI
- `.env` and Streamlit-secrets support for the Gemini key
- PowerShell helper script for saving `GEMINI_API_KEY` into the current user's environment

## What It Does

- extracts impacted areas, metadata, and appendix photos from the inspection report
- extracts page-level thermal observations, readings, thermal images, and visible companion images
- links thermal evidence to impacted areas using deterministic image similarity, with optional Gemini disambiguation
- generates a structured DDR JSON report
- renders a polished HTML report and exports it to PDF with Playwright

## Tech Stack

- `PyMuPDF` for PDF parsing and image extraction
- `Pillow` and `numpy` for image scoring
- `Pydantic` for schemas
- `google-genai` for Gemini multimodal reasoning
- `Jinja2` for HTML templating
- `Playwright` for PDF rendering

## Setup

```powershell
python -m pip install -e .[dev]
python -m playwright install chromium
```

Provider configuration can come from any of these sources:

- system environment variables such as `GEMINI_API_KEY`, `GEMINI_MODEL`, and `LLM_PROVIDER`
- local `.env` file in the repo root
- `.streamlit/secrets.toml` for the Streamlit app

Example `.env`:

```powershell
Copy-Item .env.example .env
```

Save a Gemini key at the user-environment level with:

```powershell
.\scripts\set_gemini_key.ps1
```

If no supported provider key is set, the pipeline still runs in deterministic fallback mode.

## Usage

```powershell
python -m ddr_report generate `
  --inspection "C:\path\to\Sample Report.pdf" `
  --thermal "C:\path\to\Thermal Images.pdf" `
  --out-dir "D:\report_gen\sample_outputs"
```

Disable LLM usage explicitly:

```powershell
python -m ddr_report generate --inspection <inspection.pdf> --thermal <thermal.pdf> --out-dir <output_dir> --no-llm
```

Override the API key for a single run and optionally persist it into `.env`:

```powershell
python -m ddr_report generate `
  --inspection <inspection.pdf> `
  --thermal <thermal.pdf> `
  --out-dir <output_dir> `
  --api-key "<your_key>" `
  --persist-api-key
```

## Streamlit UI

Run the upload UI with:

```powershell
streamlit run .\streamlit_app.py
```

The UI supports:

- PDF uploads for inspection and thermal documents
- Gemini provider key entry in the sidebar
- optional save-to-`.env`
- model override
- OCR toggle
- JSON, HTML, and PDF download buttons
- area-by-area report preview

## Output Layout

Each run creates a timestamped folder under the chosen output directory:

- `report.json`
- `report.html`
- `report.pdf`
- `artifacts/text/*`
- `artifacts/images/inspection/*`
- `artifacts/images/thermal/*`
- `intermediate/inspection.json`
- `intermediate/thermal.json`
- `intermediate/thermal_matches.json`
- `intermediate/evidence_bundles.json`

## Architecture Notes

- No full RAG stack is used in v1.
- Deterministic extraction happens first, then optional Gemini reasoning is applied only where it adds value.
- Uncertain thermal evidence is allowed to remain unmatched instead of being forced into an area assignment.
- Missing data is rendered as `Not Available`, and missing images as `Image Not Available`.
- The report renderer now includes thermal measurement tables and top-level summary metrics for easier review.

## Limitations

- The inspection parser is tuned for heading- and label-based reports similar to the provided sample.
- OCR fallback depends on local Tesseract support exposed through PyMuPDF.
- PDF rendering requires a local Playwright Chromium install.
- Without an API key, narrative quality is template-based rather than model-generated.
- Some thermal-to-area matches still benefit noticeably from LLM-assisted disambiguation on visually similar rooms.
- Gemini support depends on installing the `google-genai` package.
