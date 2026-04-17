from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

import fitz

from ..filesystem import RunContext
from ..models import (
    NOT_AVAILABLE,
    InspectionArea,
    InspectionDocument,
    InspectionPhoto,
    SourceRef,
)
from .common import extract_text_with_optional_ocr, normalize_multiline_text, normalize_whitespace, save_page_crop


PROPERTY_LABELS = [
    "Customer Name",
    "Mobile",
    "Email",
    "Address",
    "Property Age (In years)",
    "Floors",
    "Property Type",
]

INSPECTION_LABELS = [
    "Previous Structural audit done",
    "Previous Repair work done",
    "Inspection Date and Time",
    "Inspected By",
    "Flagged items",
    "Score",
]


def _extract_labeled_values(text: str, labels: list[str]) -> dict[str, str]:
    lines = [normalize_whitespace(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    values: dict[str, str] = {}
    label_set = set(labels)
    for idx, line in enumerate(lines):
        for label in labels:
            if line == label or line.startswith(f"{label}:") or line.startswith(f"{label} :"):
                remainder = re.sub(rf"^{re.escape(label)}\s*:?\s*", "", line).strip()
                if remainder:
                    values[label] = remainder
                else:
                    candidate = NOT_AVAILABLE
                    for next_line in lines[idx + 1 :]:
                        if next_line in label_set:
                            break
                        if any(next_line.startswith(f"{other}:") for other in labels):
                            break
                        candidate = next_line
                        break
                    values[label] = candidate or NOT_AVAILABLE
    for label in labels:
        values.setdefault(label, NOT_AVAILABLE)
    return values


def _section_between(text: str, start_pattern: str, end_pattern: str | None) -> str:
    start_match = re.search(start_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not start_match:
        return ""
    remainder = text[start_match.end() :]
    if end_pattern:
        end_match = re.search(end_pattern, remainder, flags=re.IGNORECASE | re.DOTALL)
        if end_match:
            remainder = remainder[: end_match.start()]
    return remainder


def _clean_description(value: str) -> str:
    cleaned = normalize_whitespace(value.replace("Site Details", " "))
    return cleaned or NOT_AVAILABLE


def _extract_area_photo_ids(chunk: str) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for photo_no in re.findall(r"Photo\s+(\d+)", chunk, flags=re.IGNORECASE):
        seen[f"photo-{int(photo_no):03d}"] = None
    return list(seen.keys())


def _extract_impacted_areas(doc: fitz.Document, full_text: str) -> list[InspectionArea]:
    areas: list[InspectionArea] = []
    page_texts = [extract_text_with_optional_ocr(page, enable_ocr=False) for page in doc]
    pattern = re.compile(
        r"Impacted Area\s+(\d+)(.*?)(?=Impacted Area\s+\d+|Inspection Checklists|SUMMARY TABLE|Appendix|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(full_text):
        area_number = int(match.group(1))
        chunk = match.group(2)
        negative = _section_between(chunk, r"Negative side Description", r"Negative side photographs")
        negative_photos = _section_between(chunk, r"Negative side photographs", r"Positive side Description")
        positive = _section_between(chunk, r"Positive side Description", r"Positive side photographs")
        positive_photos = _section_between(chunk, r"Positive side photographs", None)
        photo_ids = _extract_area_photo_ids(negative_photos + "\n" + positive_photos)
        page_no = next(
            (idx + 1 for idx, page_text in enumerate(page_texts) if f"Impacted Area {area_number}" in page_text),
            0,
        )
        areas.append(
            InspectionArea(
                area_id=f"area-{area_number:02d}",
                title=f"Impacted Area {area_number}",
                negative_description=_clean_description(negative),
                positive_description=_clean_description(positive),
                photo_ids=photo_ids,
                source_refs=[
                    SourceRef(
                        evidence_id=f"area-{area_number:02d}",
                        document="inspection",
                        page_no=page_no,
                        kind="impacted_area",
                        label=f"Impacted Area {area_number}",
                    )
                ],
            )
        )
    return areas


def _extract_summary_points(full_text: str) -> list[str]:
    match = re.search(r"SUMMARY TABLE(.*?)(?:Appendix|$)", full_text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    lines = [normalize_whitespace(line) for line in match.group(1).splitlines()]
    return [line for line in lines if line.startswith("Observed ")]


def _appendix_start_page(doc: fitz.Document) -> int:
    for idx, page in enumerate(doc):
        text = page.get_text("text")
        if "Appendix" in text:
            return idx
    for idx, page in enumerate(doc):
        text = page.get_text("text")
        image_count = len(page.get_images(full=True))
        if re.search(r"Photo\s+\d+", text) and image_count >= 4:
            return idx
    return max(doc.page_count - 1, 0)


def _pair_appendix_images(doc: fitz.Document, run_context: RunContext) -> list[InspectionPhoto]:
    photos: list[InspectionPhoto] = []
    appendix_start = _appendix_start_page(doc)
    seen_ids: set[str] = set()
    for page_idx in range(appendix_start, doc.page_count):
        page = doc.load_page(page_idx)
        page_dict = page.get_text("dict")
        image_blocks = [
            block
            for block in page_dict["blocks"]
            if block.get("type") == 1
            and block.get("bbox", (0, 0, 0, 0))[1] > 80
            and block.get("width", 0) > 140
            and block.get("height", 0) > 140
        ]
        image_blocks.sort(key=lambda block: (round(block["bbox"][1], 1), round(block["bbox"][0], 1)))
        caption_lines: list[tuple[int, tuple[float, float, float, float]]] = []
        for block in page_dict["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                text = normalize_whitespace(text)
                match = re.search(r"Photo\s+(\d+)", text, flags=re.IGNORECASE)
                if match:
                    caption_lines.append((int(match.group(1)), tuple(line.get("bbox", block.get("bbox", (0, 0, 0, 0))))))

        page_mid_x = float(page.rect.width) / 2

        def column_from_bbox(bbox: tuple[float, float, float, float]) -> int:
            center_x = (bbox[0] + bbox[2]) / 2
            return 0 if center_x < page_mid_x else 1

        captions_by_col: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {0: [], 1: []}
        for photo_no, bbox in caption_lines:
            captions_by_col[column_from_bbox(bbox)].append((photo_no, bbox))
        for items in captions_by_col.values():
            items.sort(key=lambda item: (item[1][1], item[1][0]))

        images_by_col: dict[int, list[dict]] = {0: [], 1: []}
        for block in image_blocks:
            images_by_col[column_from_bbox(tuple(block["bbox"]))].append(block)
        for items in images_by_col.values():
            items.sort(key=lambda block: (block["bbox"][1], block["bbox"][0]))

        assignments: list[tuple[int, dict]] = []
        for column in (0, 1):
            available = captions_by_col[column][:]
            for image_block in images_by_col[column]:
                image_bbox = tuple(image_block["bbox"])
                image_center_y = (image_bbox[1] + image_bbox[3]) / 2
                best_index = None
                best_cost = None
                for idx, (photo_no, caption_bbox) in enumerate(available):
                    caption_center_y = (caption_bbox[1] + caption_bbox[3]) / 2
                    below_penalty = 30.0 if caption_center_y > image_center_y else 0.0
                    cost = abs(image_center_y - caption_center_y) + below_penalty
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_index = idx
                if best_index is None:
                    continue
                photo_no, _ = available.pop(best_index)
                assignments.append((photo_no, image_block))

        for photo_no, image_block in sorted(assignments, key=lambda item: item[0]):
            photo_id = f"photo-{photo_no:03d}"
            if photo_id in seen_ids:
                continue
            image_path = save_page_crop(
                page,
                tuple(image_block["bbox"]),
                run_context.inspection_images_dir / photo_id,
            )
            seen_ids.add(photo_id)
            photos.append(
                InspectionPhoto(
                    photo_id=photo_id,
                    caption_ref=f"Photo {photo_no}",
                    image_path=image_path,
                    page_no=page_idx + 1,
                    source_refs=[
                        SourceRef(
                            evidence_id=photo_id,
                            document="inspection",
                            page_no=page_idx + 1,
                            kind="photo",
                            label=f"Photo {photo_no}",
                        )
                    ],
                )
            )
    photos.sort(key=lambda photo: photo.photo_id)
    return photos


def _backfill_area_photo_ids(areas: list[InspectionArea], photos: list[InspectionPhoto]) -> None:
    photo_numbers = sorted(int(photo.photo_id.split("-")[-1]) for photo in photos)
    observed_numbers = {
        area.area_id: [int(photo_id.split("-")[-1]) for photo_id in area.photo_ids]
        for area in areas
    }
    assigned: set[int] = {number for numbers in observed_numbers.values() for number in numbers}
    unassigned = [number for number in photo_numbers if number not in assigned]
    if not unassigned:
        return

    for idx, area in enumerate(areas):
        current = observed_numbers[area.area_id]
        next_min = None
        for next_area in areas[idx + 1 :]:
            next_numbers = observed_numbers[next_area.area_id]
            if next_numbers:
                next_min = min(next_numbers)
                break
        if current:
            lower_bound = max(current) + 1
        else:
            previous_max = 0
            for previous_area in areas[:idx]:
                previous_numbers = observed_numbers[previous_area.area_id]
                if previous_numbers:
                    previous_max = max(previous_max, max(previous_numbers))
            lower_bound = previous_max + 1
        upper_bound = (next_min - 1) if next_min else photo_numbers[-1]
        inferred = [number for number in unassigned if lower_bound <= number <= upper_bound]
        for number in inferred:
            photo_id = f"photo-{number:03d}"
            if photo_id not in area.photo_ids:
                area.photo_ids.append(photo_id)


def extract_inspection_report(path: Path, run_context: RunContext, enable_ocr: bool = False) -> InspectionDocument:
    doc = fitz.open(path)
    page_texts: list[str] = []
    for idx, page in enumerate(doc):
        page_text = extract_text_with_optional_ocr(page, enable_ocr=enable_ocr)
        page_texts.append(page_text)
        (run_context.extracted_text_dir / f"inspection_page_{idx + 1:03d}.txt").write_text(page_text, encoding="utf-8")

    full_text = normalize_multiline_text("\n".join(page_texts))
    property_metadata = _extract_labeled_values(full_text, PROPERTY_LABELS)
    inspection_metadata = _extract_labeled_values(full_text, INSPECTION_LABELS)
    areas = _extract_impacted_areas(doc, full_text)
    photos = _pair_appendix_images(doc, run_context)
    _backfill_area_photo_ids(areas, photos)
    source_index: list[SourceRef] = []
    for area in areas:
        source_index.extend(area.source_refs)
    for photo in photos:
        source_index.extend(photo.source_refs)

    return InspectionDocument(
        document_path=str(path),
        property_metadata=property_metadata,
        inspection_metadata=inspection_metadata,
        impacted_areas=areas,
        photos=photos,
        auxiliary_summary_points=_extract_summary_points(full_text),
        source_index=source_index,
    )
