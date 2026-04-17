from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.sync_api import sync_playwright

from .extraction.common import data_uri_for_file
from .filesystem import RunContext
from .models import DDRReport, IMAGE_NOT_AVAILABLE, InspectionDocument, ThermalDocument


def _image_entries(paths: list[str]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for index, path in enumerate(paths, start=1):
        if path == IMAGE_NOT_AVAILABLE:
            continue
        file_path = Path(path)
        if not file_path.exists():
            continue
        entries.append(
            {
                "label": f"Image {index}",
                "data_uri": data_uri_for_file(file_path),
                "filename": file_path.name,
            }
        )
    return entries


def _build_template_context(report: DDRReport, inspection_doc: InspectionDocument, thermal_doc: ThermalDocument) -> dict:
    thermal_lookup = {observation.obs_id: observation for observation in thermal_doc.observations}
    matched_count = sum(
        1 for area in report.area_wise_observations for path in area.thermal_image_paths if path != IMAGE_NOT_AVAILABLE
    )
    area_cards = []
    for area in report.area_wise_observations:
        area_cards.append(
            {
                "title": area.area_title,
                "summary": area.observation_summary,
                "root_cause": area.probable_root_cause,
                "severity": area.severity,
                "severity_reasoning": area.severity_reasoning,
                "recommended_actions": area.recommended_actions,
                "additional_notes": area.additional_notes,
                "missing_items": area.missing_or_unclear_information,
                "inspection_images": _image_entries(area.inspection_photo_paths),
                "thermal_images": _image_entries(area.thermal_image_paths),
                "visible_images": _image_entries(area.visible_image_paths),
                "thermal_measurements": area.thermal_measurements,
                "evidence_refs": area.evidence_refs,
            }
        )

    unmatched_cards = []
    for item in report.unmatched_thermal_observations:
        observation = thermal_lookup.get(item.obs_id)
        unmatched_cards.append(
            {
                "obs_id": item.obs_id,
                "confidence": f"{item.confidence:.2f}",
                "rationale": item.rationale,
                "candidates": item.candidate_scores,
                "thermal_images": _image_entries([observation.thermal_image_path] if observation else []),
                "visible_images": _image_entries([observation.visible_image_path] if observation else []),
            }
        )

    metadata = {
        **inspection_doc.property_metadata,
        **inspection_doc.inspection_metadata,
        "Inspection Source": Path(inspection_doc.document_path).name,
        "Thermal Source": Path(thermal_doc.document_path).name,
    }

    return {
        "report": report,
        "area_cards": area_cards,
        "unmatched_cards": unmatched_cards,
        "metadata": metadata,
        "summary_metrics": {
            "impacted_area_count": len(report.area_wise_observations),
            "matched_thermal_count": matched_count,
            "unmatched_thermal_count": len(report.unmatched_thermal_observations),
            "inspection_image_count": len(inspection_doc.photos),
        },
    }


def render_report_html(
    report: DDRReport,
    inspection_doc: InspectionDocument,
    thermal_doc: ThermalDocument,
    run_context: RunContext,
) -> Path:
    templates_dir = Path(__file__).with_name("templates")
    environment = Environment(
        loader=FileSystemLoader(templates_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = environment.get_template("report.html.j2")
    html = template.render(**_build_template_context(report, inspection_doc, thermal_doc))
    run_context.report_html_path.write_text(html, encoding="utf-8")
    return run_context.report_html_path


def render_report_pdf(html_path: Path, pdf_path: Path) -> Path:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.resolve().as_uri(), wait_until="load")
            page.emulate_media(media="screen")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "14mm", "right": "12mm", "bottom": "14mm", "left": "12mm"},
            )
            browser.close()
    except Exception as exc:
        raise RuntimeError(
            "Failed to render PDF with Playwright. Ensure Chromium is installed via "
            "'python -m playwright install chromium'."
        ) from exc
    return pdf_path
