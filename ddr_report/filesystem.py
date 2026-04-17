from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class RunContext:
    root: Path
    extracted_text_dir: Path
    images_dir: Path
    inspection_images_dir: Path
    thermal_images_dir: Path
    intermediate_dir: Path
    report_json_path: Path
    report_html_path: Path
    report_pdf_path: Path
    inspection_json_path: Path
    thermal_json_path: Path
    bundle_json_path: Path


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return value or "report"


def prepare_run_context(out_dir: Path, inspection_path: Path, thermal_path: Path) -> RunContext:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{slugify(inspection_path.stem)}__{slugify(thermal_path.stem)}__{timestamp}"
    root = out_dir / run_name
    extracted_text_dir = root / "artifacts" / "text"
    images_dir = root / "artifacts" / "images"
    inspection_images_dir = images_dir / "inspection"
    thermal_images_dir = images_dir / "thermal"
    intermediate_dir = root / "intermediate"

    for path in (
        extracted_text_dir,
        inspection_images_dir,
        thermal_images_dir,
        intermediate_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    return RunContext(
        root=root,
        extracted_text_dir=extracted_text_dir,
        images_dir=images_dir,
        inspection_images_dir=inspection_images_dir,
        thermal_images_dir=thermal_images_dir,
        intermediate_dir=intermediate_dir,
        report_json_path=root / "report.json",
        report_html_path=root / "report.html",
        report_pdf_path=root / "report.pdf",
        inspection_json_path=intermediate_dir / "inspection.json",
        thermal_json_path=intermediate_dir / "thermal.json",
        bundle_json_path=intermediate_dir / "evidence_bundles.json",
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
