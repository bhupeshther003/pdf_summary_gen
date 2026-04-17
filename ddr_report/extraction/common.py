from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Iterable

import fitz


def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_multiline_text(value: str) -> str:
    lines = [normalize_whitespace(line) for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def save_image_bytes(image_bytes: bytes, ext: str, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != f".{ext.lower()}":
        output_path = output_path.with_suffix(f".{ext.lower()}")
    output_path.write_bytes(image_bytes)
    return str(output_path)


def save_page_crop(page: fitz.Page, bbox: tuple[float, float, float, float], output_path: Path, zoom: float = 2.0) -> str:
    output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rect = fitz.Rect(bbox)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
    pixmap.save(output_path)
    return str(output_path)


def data_uri_for_file(path: str | Path) -> str:
    file_path = Path(path)
    mime = "image/png"
    if file_path.suffix.lower() in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def extract_text_with_optional_ocr(page: fitz.Page, enable_ocr: bool) -> str:
    text = page.get_text("text")
    if normalize_whitespace(text):
        return text
    if not enable_ocr:
        return text
    try:
        text_page = page.get_textpage_ocr()
        return page.get_text("text", textpage=text_page)
    except Exception:
        return text


def collect_text_lines(blocks: Iterable[dict]) -> list[tuple[tuple[float, float, float, float], str]]:
    lines: list[tuple[tuple[float, float, float, float], str]] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        bbox = tuple(block.get("bbox", (0.0, 0.0, 0.0, 0.0)))
        fragments: list[str] = []
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = normalize_whitespace(text)
            if text:
                fragments.append(text)
        if fragments:
            lines.append((bbox, " ".join(fragments)))
    return lines
