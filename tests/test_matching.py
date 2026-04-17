from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image

from ddr_report.matching import assign_thermal_observations, score_area_candidates
from ddr_report.models import InspectionArea, InspectionPhoto, ThermalObservation


def _save_image(path: Path, color: tuple[int, int, int]) -> str:
    image = Image.new("RGB", (420, 320), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return str(path)


def test_score_area_candidates_prefers_similar_images(tmp_path: Path) -> None:
    photo_a = InspectionPhoto(photo_id="photo-001", caption_ref="Photo 1", image_path=_save_image(tmp_path / "a.png", (180, 60, 60)), page_no=1)
    photo_b = InspectionPhoto(photo_id="photo-002", caption_ref="Photo 2", image_path=_save_image(tmp_path / "b.png", (60, 60, 180)), page_no=1)
    areas = [
        InspectionArea(area_id="area-01", title="Impacted Area 1", photo_ids=["photo-001"]),
        InspectionArea(area_id="area-02", title="Impacted Area 2", photo_ids=["photo-002"]),
    ]
    observation = ThermalObservation(
        obs_id="thermal-obs-001",
        page_no=1,
        image_name="test",
        visible_image_path=_save_image(tmp_path / "probe.png", (175, 65, 65)),
    )
    candidates = score_area_candidates(observation, areas, {photo.photo_id: photo for photo in [photo_a, photo_b]})

    assert candidates[0].area_id == "area-01"
    assert candidates[0].score > candidates[1].score


def test_assign_thermal_observations_matches_best_area(tmp_path: Path) -> None:
    photo_a = InspectionPhoto(photo_id="photo-001", caption_ref="Photo 1", image_path=_save_image(tmp_path / "a.png", (180, 60, 60)), page_no=1)
    photo_b = InspectionPhoto(photo_id="photo-002", caption_ref="Photo 2", image_path=_save_image(tmp_path / "b.png", (60, 60, 180)), page_no=1)
    areas = [
        InspectionArea(area_id="area-01", title="Impacted Area 1", photo_ids=["photo-001"]),
        InspectionArea(area_id="area-02", title="Impacted Area 2", photo_ids=["photo-002"]),
    ]
    observation = ThermalObservation(
        obs_id="thermal-obs-001",
        page_no=1,
        image_name="test",
        visible_image_path=_save_image(tmp_path / "probe.png", (175, 65, 65)),
    )

    decisions = assign_thermal_observations([observation], areas, [photo_a, photo_b], reasoner=None)

    assert decisions[0].matched_area_id == "area-01"
