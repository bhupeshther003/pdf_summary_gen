from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


NOT_AVAILABLE = "Not Available"
IMAGE_NOT_AVAILABLE = "Image Not Available"


class SourceRef(BaseModel):
    evidence_id: str
    document: Literal["inspection", "thermal"]
    page_no: int
    kind: str
    label: str | None = None


class InspectionPhoto(BaseModel):
    photo_id: str
    caption_ref: str
    image_path: str
    page_no: int
    source_refs: list[SourceRef] = Field(default_factory=list)


class InspectionArea(BaseModel):
    area_id: str
    title: str
    negative_description: str = NOT_AVAILABLE
    positive_description: str = NOT_AVAILABLE
    photo_ids: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


class InspectionDocument(BaseModel):
    document_path: str
    property_metadata: dict[str, str] = Field(default_factory=dict)
    inspection_metadata: dict[str, str] = Field(default_factory=dict)
    impacted_areas: list[InspectionArea] = Field(default_factory=list)
    photos: list[InspectionPhoto] = Field(default_factory=list)
    auxiliary_summary_points: list[str] = Field(default_factory=list)
    source_index: list[SourceRef] = Field(default_factory=list)


class ThermalObservation(BaseModel):
    obs_id: str
    page_no: int
    image_name: str = NOT_AVAILABLE
    date: str = NOT_AVAILABLE
    hotspot_c: float | None = None
    coldspot_c: float | None = None
    emissivity: float | None = None
    reflected_temp_c: float | None = None
    thermal_image_path: str = IMAGE_NOT_AVAILABLE
    visible_image_path: str = IMAGE_NOT_AVAILABLE
    source_refs: list[SourceRef] = Field(default_factory=list)


class ThermalDocument(BaseModel):
    document_path: str
    observations: list[ThermalObservation] = Field(default_factory=list)
    source_index: list[SourceRef] = Field(default_factory=list)


class CandidateMatch(BaseModel):
    area_id: str
    area_title: str
    score: float
    supporting_photo_ids: list[str] = Field(default_factory=list)


class ThermalMatchDecision(BaseModel):
    obs_id: str
    matched_area_id: str
    confidence: float
    rationale: str
    candidate_scores: list[CandidateMatch] = Field(default_factory=list)
    method: Literal["heuristic", "llm", "fallback"] = "heuristic"


class MeasurementSummary(BaseModel):
    obs_id: str
    date: str = NOT_AVAILABLE
    hotspot_c: float | None = None
    coldspot_c: float | None = None
    reflected_temp_c: float | None = None
    emissivity: float | None = None


class EvidenceBundle(BaseModel):
    area_id: str
    area_title: str
    negative_description: str = NOT_AVAILABLE
    positive_description: str = NOT_AVAILABLE
    inspection_refs: list[str] = Field(default_factory=list)
    thermal_refs: list[str] = Field(default_factory=list)
    inspection_photos: list[InspectionPhoto] = Field(default_factory=list)
    thermal_observations: list[ThermalObservation] = Field(default_factory=list)
    measurements: list[MeasurementSummary] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    missing_items: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)


class SeveritySummary(BaseModel):
    area_id: str
    area_title: str
    severity: Literal["Low", "Moderate", "High"]
    reasoning: str


class DDRAreaObservation(BaseModel):
    area_id: str
    area_title: str
    observation_summary: str
    probable_root_cause: str
    severity: Literal["Low", "Moderate", "High"]
    severity_reasoning: str
    recommended_actions: list[str] = Field(default_factory=list)
    additional_notes: list[str] = Field(default_factory=list)
    missing_or_unclear_information: list[str] = Field(default_factory=list)
    inspection_photo_paths: list[str] = Field(default_factory=list)
    thermal_image_paths: list[str] = Field(default_factory=list)
    visible_image_paths: list[str] = Field(default_factory=list)
    thermal_measurements: list[MeasurementSummary] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class SourceIndexEntry(BaseModel):
    evidence_id: str
    document: str
    page_no: int
    kind: str
    label: str | None = None


class DDRReport(BaseModel):
    property_issue_summary: list[str] = Field(default_factory=list)
    area_wise_observations: list[DDRAreaObservation] = Field(default_factory=list)
    probable_root_cause: list[str] = Field(default_factory=list)
    severity_assessment: list[SeveritySummary] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    additional_notes: list[str] = Field(default_factory=list)
    missing_or_unclear_information: list[str] = Field(default_factory=list)
    source_index: list[SourceIndexEntry] = Field(default_factory=list)
    unmatched_thermal_observations: list[ThermalMatchDecision] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z")
    llm_mode: str = "disabled"


class PipelineResult(BaseModel):
    run_dir: str
    report_json_path: str
    report_html_path: str
    report_pdf_path: str
    inspection_json_path: str
    thermal_json_path: str
    bundle_json_path: str
