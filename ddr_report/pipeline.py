from __future__ import annotations

from pathlib import Path

from .config import Settings
from .extraction.inspection import extract_inspection_report
from .extraction.thermal import extract_thermal_report
from .filesystem import prepare_run_context, write_json
from .llm import build_reasoner
from .matching import assign_thermal_observations
from .models import PipelineResult
from .rendering import render_report_html, render_report_pdf
from .synthesis import build_ddr_report, build_evidence_bundles


def run_pipeline(
    inspection_path: str | Path,
    thermal_path: str | Path,
    out_dir: str | Path,
    settings: Settings | None = None,
) -> PipelineResult:
    settings = settings or Settings.from_env()
    inspection_path = Path(inspection_path)
    thermal_path = Path(thermal_path)
    out_dir = Path(out_dir)
    run_context = prepare_run_context(out_dir, inspection_path, thermal_path)

    inspection_doc = extract_inspection_report(inspection_path, run_context, enable_ocr=settings.enable_ocr)
    thermal_doc = extract_thermal_report(thermal_path, run_context, enable_ocr=settings.enable_ocr)
    write_json(run_context.inspection_json_path, inspection_doc.model_dump(mode="json"))
    write_json(run_context.thermal_json_path, thermal_doc.model_dump(mode="json"))

    reasoner = None
    if settings.llm_enabled:
        try:
            reasoner = build_reasoner(settings)
        except Exception:
            reasoner = None

    decisions = assign_thermal_observations(
        thermal_doc.observations,
        inspection_doc.impacted_areas,
        inspection_doc.photos,
        reasoner=reasoner,
    )
    write_json(run_context.intermediate_dir / "thermal_matches.json", [decision.model_dump(mode="json") for decision in decisions])

    bundles, unmatched = build_evidence_bundles(inspection_doc, thermal_doc, decisions)
    write_json(run_context.bundle_json_path, [bundle.model_dump(mode="json") for bundle in bundles])

    report = build_ddr_report(inspection_doc, thermal_doc, bundles, unmatched, reasoner=reasoner)
    write_json(run_context.report_json_path, report.model_dump(mode="json"))
    render_report_html(report, inspection_doc, thermal_doc, run_context)
    render_report_pdf(run_context.report_html_path, run_context.report_pdf_path)

    return PipelineResult(
        run_dir=str(run_context.root),
        report_json_path=str(run_context.report_json_path),
        report_html_path=str(run_context.report_html_path),
        report_pdf_path=str(run_context.report_pdf_path),
        inspection_json_path=str(run_context.inspection_json_path),
        thermal_json_path=str(run_context.thermal_json_path),
        bundle_json_path=str(run_context.bundle_json_path),
    )
