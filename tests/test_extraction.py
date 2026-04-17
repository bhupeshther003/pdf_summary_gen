from __future__ import annotations

from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image

from ddr_report.extraction.inspection import extract_inspection_report
from ddr_report.extraction.thermal import extract_thermal_report
from ddr_report.filesystem import prepare_run_context


def _image_bytes(color: tuple[int, int, int], size: tuple[int, int] = (420, 320)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _make_inspection_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(40, 40, 540, 290),
        (
            "Inspection Form\n"
            "Customer Name: Jane Doe\n"
            "Property Type: Flat\n"
            "Inspection Date and Time: 01.01.2026 10:00 IST\n"
            "Impacted Area 1\n"
            "Negative side Description\n"
            "Hall Skirting level Dampness\n"
            "Negative side photographs\n"
            "Photo 1\n"
            "Positive side Description\n"
            "Bathroom tile hollowness\n"
            "Positive side photographs\n"
            "Photo 2\n"
        ),
        fontsize=12,
    )

    appendix = doc.new_page(width=595, height=842)
    appendix.insert_text((40, 40), "Appendix", fontsize=16)
    img1 = _image_bytes((180, 80, 80))
    img2 = _image_bytes((80, 120, 180))
    appendix.insert_image(fitz.Rect(70, 90, 255, 255), stream=img1)
    appendix.insert_text((70, 275), "Photo 1", fontsize=12)
    appendix.insert_image(fitz.Rect(315, 90, 500, 255), stream=img2)
    appendix.insert_text((315, 275), "Photo 2", fontsize=12)
    doc.save(path)
    doc.close()


def _make_thermal_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_textbox(
        fitz.Rect(340, 90, 560, 280),
        (
            "27/09/22\n"
            "Hotspot : 28.8 C\n"
            "Coldspot : 23.4 C\n"
            "Emissivity : 0.94\n"
            "Reflected temperature : 23 C\n"
            "Thermal image : SAMPLE001.JPG\n"
            "Device : GTC 400 C Professional\n"
        ),
        fontsize=12,
    )
    page.insert_image(fitz.Rect(30, 80, 300, 320), stream=_image_bytes((210, 60, 60), (640, 480)))
    page.insert_image(fitz.Rect(30, 360, 300, 600), stream=_image_bytes((80, 80, 210), (640, 480)))
    doc.save(path)
    doc.close()


def test_extract_inspection_report(tmp_path: Path) -> None:
    inspection_pdf = tmp_path / "inspection.pdf"
    thermal_pdf = tmp_path / "thermal.pdf"
    _make_inspection_pdf(inspection_pdf)
    _make_thermal_pdf(thermal_pdf)
    run_context = prepare_run_context(tmp_path / "out", inspection_pdf, thermal_pdf)

    report = extract_inspection_report(inspection_pdf, run_context)

    assert report.property_metadata["Customer Name"] == "Jane Doe"
    assert len(report.impacted_areas) == 1
    assert report.impacted_areas[0].negative_description == "Hall Skirting level Dampness"
    assert report.impacted_areas[0].photo_ids == ["photo-001", "photo-002"]
    assert len(report.photos) == 2


def test_extract_thermal_report(tmp_path: Path) -> None:
    inspection_pdf = tmp_path / "inspection.pdf"
    thermal_pdf = tmp_path / "thermal.pdf"
    _make_inspection_pdf(inspection_pdf)
    _make_thermal_pdf(thermal_pdf)
    run_context = prepare_run_context(tmp_path / "out", inspection_pdf, thermal_pdf)

    report = extract_thermal_report(thermal_pdf, run_context)

    assert len(report.observations) == 1
    observation = report.observations[0]
    assert observation.image_name == "SAMPLE001.JPG"
    assert observation.hotspot_c == 28.8
    assert observation.coldspot_c == 23.4
    assert Path(observation.thermal_image_path).exists()
    assert Path(observation.visible_image_path).exists()
