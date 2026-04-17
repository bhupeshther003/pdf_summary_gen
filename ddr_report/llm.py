from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .config import Settings
from .extraction.common import data_uri_for_file
from .models import CandidateMatch, EvidenceBundle, ThermalMatchDecision, ThermalObservation


class LLMThermalMatch(BaseModel):
    matched_area_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class LLMAreaNarrative(BaseModel):
    observation_summary: str
    probable_root_cause: str
    severity: str
    severity_reasoning: str
    recommended_actions: list[str]
    additional_notes: list[str]
    missing_or_unclear_information: list[str]


class LLMGlobalNarrative(BaseModel):
    property_issue_summary: list[str]
    probable_root_cause: list[str]
    recommended_actions: list[str]
    additional_notes: list[str]
    missing_or_unclear_information: list[str]


class BaseReasoner:
    provider = "unknown"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @property
    def llm_mode(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def resolve_thermal_match(
        self,
        observation: ThermalObservation,
        candidates: list[CandidateMatch],
        candidate_payload: list[dict[str, Any]],
    ) -> ThermalMatchDecision | None:
        raise NotImplementedError

    def narrate_area_bundle(self, bundle: EvidenceBundle) -> LLMAreaNarrative | None:
        raise NotImplementedError

    def narrate_global_sections(
        self,
        bundles: list[EvidenceBundle],
        unmatched: list[ThermalMatchDecision],
        llm_mode: str,
    ) -> LLMGlobalNarrative | None:
        raise NotImplementedError


def _response_text(response: Any) -> str | None:
    text = getattr(response, "text", None)
    if text:
        return str(text)

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        chunks = [str(part.text) for part in parts if getattr(part, "text", None)]
        if chunks:
            return "".join(chunks)
    return None


def _image_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/png"


class GeminiReasoner(BaseReasoner):
    provider = "gemini"

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required to initialize GeminiReasoner")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError("google-genai is not installed. Run `pip install google-genai`.") from exc

        super().__init__(settings.gemini_model)
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._types = types

    def _image_part(self, image_path: str) -> Any | None:
        path = Path(image_path)
        if not path.exists():
            return None
        return self._types.Part.from_bytes(data=path.read_bytes(), mime_type=_image_mime_type(path))

    def _generate_json(
        self,
        prompt_parts: list[Any],
        response_model: type[BaseModel],
        temperature: float,
    ) -> BaseModel | None:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt_parts,
            config={
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
            },
        )
        text = _response_text(response)
        if not text:
            return None
        return response_model.model_validate_json(text)

    def resolve_thermal_match(
        self,
        observation: ThermalObservation,
        candidates: list[CandidateMatch],
        candidate_payload: list[dict[str, Any]],
    ) -> ThermalMatchDecision | None:
        if not candidates:
            return None

        content: list[Any] = [
            (
                "Match this thermal observation to exactly one impacted area from the candidate list "
                "or return 'unmatched' if the evidence is weak. Use only the provided evidence. "
                "Never invent facts.\n\n"
                f"Thermal observation:\n{json.dumps(observation.model_dump(), indent=2)}\n\n"
                f"Candidate areas:\n{json.dumps(candidate_payload, indent=2)}\n\n"
                "Return JSON only."
            )
        ]
        visible_part = self._image_part(observation.visible_image_path)
        if visible_part is not None:
            content.extend(["Thermal observation visible-light image:", visible_part])
        thermal_part = self._image_part(observation.thermal_image_path)
        if thermal_part is not None:
            content.extend(["Thermal observation thermal image:", thermal_part])
        for payload in candidate_payload:
            for image_path in payload.get("supporting_images", [])[:1]:
                image_part = self._image_part(image_path)
                if image_part is not None:
                    content.extend([f"Candidate area {payload['area_id']} support image:", image_part])

        parsed = self._generate_json(content, LLMThermalMatch, temperature=0.0)
        if not parsed:
            return None
        matched_area_id = parsed.matched_area_id
        if matched_area_id not in {candidate.area_id for candidate in candidates} | {"unmatched"}:
            matched_area_id = "unmatched"
        return ThermalMatchDecision(
            obs_id=observation.obs_id,
            matched_area_id=matched_area_id,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            candidate_scores=candidates,
            method="llm",
        )

    def narrate_area_bundle(self, bundle: EvidenceBundle) -> LLMAreaNarrative | None:
        payload = {
            "bundle": bundle.model_dump(mode="json"),
            "instructions": {
                "missing_value": "Not Available",
                "image_missing_value": "Image Not Available",
            },
        }
        return self._generate_json(
            [
                (
                    "Write client-friendly DDR content for a single impacted area. "
                    "Use only the supplied evidence. Keep language simple, factual, and non-speculative.\n\n"
                    f"{json.dumps(payload, indent=2)}"
                )
            ],
            LLMAreaNarrative,
            temperature=0.2,
        )

    def narrate_global_sections(
        self,
        bundles: list[EvidenceBundle],
        unmatched: list[ThermalMatchDecision],
        llm_mode: str,
    ) -> LLMGlobalNarrative | None:
        payload = {
            "bundles": [bundle.model_dump(mode="json") for bundle in bundles],
            "unmatched_thermal_observations": [decision.model_dump(mode="json") for decision in unmatched],
            "llm_mode": llm_mode,
        }
        return self._generate_json(
            [
                (
                    "Write the overall DDR sections for a property report. "
                    "Use concise, client-friendly language. Do not invent facts, "
                    "and keep missing data explicit as 'Not Available'.\n\n"
                    f"{json.dumps(payload, indent=2)}"
                )
            ],
            LLMGlobalNarrative,
            temperature=0.2,
        )


def build_reasoner(settings: Settings) -> BaseReasoner | None:
    if not settings.llm_enabled:
        return None
    if settings.llm_provider == "gemini" and settings.gemini_api_key:
        return GeminiReasoner(settings)
    return None
