from __future__ import annotations

import re
from pathlib import Path

import fitz

from ..filesystem import RunContext
from ..models import IMAGE_NOT_AVAILABLE, NOT_AVAILABLE, SourceRef, ThermalDocument, ThermalObservation
from .common import extract_float, extract_text_with_optional_ocr, normalize_multiline_text, save_page_crop


def _extract_image_name(text: str) -> str:
    match = re.search(r"Thermal image\s*:\s*([^\n]+?)(?:\s+Device\s*:|\s+Device\s+|$)", text, flags=re.IGNORECASE)
    if not match:
        return NOT_AVAILABLE
    return match.group(1).strip()


def _extract_date(text: str) -> str:
    match = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", text)
    return match.group(0) if match else NOT_AVAILABLE


def extract_thermal_report(path: Path, run_context: RunContext, enable_ocr: bool = False) -> ThermalDocument:
    doc = fitz.open(path)
    observations: list[ThermalObservation] = []
    source_index: list[SourceRef] = []
    for idx, page in enumerate(doc):
        page_no = idx + 1
        page_text = extract_text_with_optional_ocr(page, enable_ocr=enable_ocr)
        (run_context.extracted_text_dir / f"thermal_page_{page_no:03d}.txt").write_text(page_text, encoding="utf-8")

        page_dict = page.get_text("dict")
        image_blocks = [
            block
            for block in page_dict["blocks"]
            if block.get("type") == 1 and block.get("width", 0) >= 300 and block.get("height", 0) >= 200
        ]
        image_blocks.sort(key=lambda block: (round(block["bbox"][1], 1), round(block["bbox"][0], 1)))

        obs_id = f"thermal-obs-{page_no:03d}"
        thermal_image_path = IMAGE_NOT_AVAILABLE
        visible_image_path = IMAGE_NOT_AVAILABLE
        if image_blocks:
            thermal_block = image_blocks[0]
            thermal_image_path = save_page_crop(
                page,
                tuple(thermal_block["bbox"]),
                run_context.thermal_images_dir / f"{obs_id}_thermal",
            )
        if len(image_blocks) > 1:
            visible_block = image_blocks[1]
            visible_image_path = save_page_crop(
                page,
                tuple(visible_block["bbox"]),
                run_context.thermal_images_dir / f"{obs_id}_visible",
            )

        normalized_text = normalize_multiline_text(page_text)
        observation = ThermalObservation(
            obs_id=obs_id,
            page_no=page_no,
            image_name=_extract_image_name(normalized_text),
            date=_extract_date(normalized_text),
            hotspot_c=extract_float(r"Hotspot\s*:\s*([0-9.]+)", normalized_text),
            coldspot_c=extract_float(r"Coldspot\s*:\s*([0-9.]+)", normalized_text),
            emissivity=extract_float(r"Emissivity\s*:\s*([0-9.]+)", normalized_text),
            reflected_temp_c=extract_float(r"Reflected temperature\s*:\s*([0-9.]+)", normalized_text),
            thermal_image_path=thermal_image_path,
            visible_image_path=visible_image_path,
            source_refs=[
                SourceRef(
                    evidence_id=obs_id,
                    document="thermal",
                    page_no=page_no,
                    kind="thermal_observation",
                    label=f"Thermal page {page_no}",
                )
            ],
        )
        observations.append(observation)
        source_index.extend(observation.source_refs)

    return ThermalDocument(document_path=str(path), observations=observations, source_index=source_index)
