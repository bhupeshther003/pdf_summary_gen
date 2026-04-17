from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .llm import BaseReasoner
from .models import CandidateMatch, InspectionArea, InspectionPhoto, ThermalMatchDecision, ThermalObservation


def _fingerprint(image_path: str) -> np.ndarray | None:
    path = Path(image_path)
    if not path.exists():
        return None
    image = Image.open(path).convert("RGB").resize((32, 32))
    array = np.asarray(image, dtype=np.float32) / 255.0
    vector = array.reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector / norm


def _similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def score_area_candidates(
    observation: ThermalObservation,
    areas: list[InspectionArea],
    photo_lookup: dict[str, InspectionPhoto],
) -> list[CandidateMatch]:
    probe_path = observation.visible_image_path
    if not Path(probe_path).exists():
        probe_path = observation.thermal_image_path
    probe = _fingerprint(probe_path)
    candidates: list[CandidateMatch] = []
    for area in areas:
        scored: list[tuple[float, str]] = []
        for photo_id in area.photo_ids:
            photo = photo_lookup.get(photo_id)
            if not photo:
                continue
            score = _similarity(probe, _fingerprint(photo.image_path))
            scored.append((score, photo_id))
        scored.sort(reverse=True)
        if scored:
            candidates.append(
                CandidateMatch(
                    area_id=area.area_id,
                    area_title=area.title,
                    score=scored[0][0],
                    supporting_photo_ids=[photo_id for _, photo_id in scored[:3]],
                )
            )
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def _heuristic_decision(observation: ThermalObservation, candidates: list[CandidateMatch]) -> ThermalMatchDecision:
    if not candidates:
        return ThermalMatchDecision(
            obs_id=observation.obs_id,
            matched_area_id="unmatched",
            confidence=0.0,
            rationale="No inspection images were available for deterministic matching.",
            candidate_scores=[],
            method="fallback",
        )

    best = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else -1.0
    score_gap = best.score - second_score
    confident = (best.score >= 0.96 and score_gap >= 0.008) or (best.score >= 0.98 and score_gap >= 0.004)
    if confident:
        return ThermalMatchDecision(
            obs_id=observation.obs_id,
            matched_area_id=best.area_id,
            confidence=min(0.93, max(0.62, 0.55 + (score_gap * 12))),
            rationale=(
                f"Deterministic image similarity favored {best.area_title} with score {best.score:.2f}. "
                f"Runner-up score was {second_score:.2f}, which left a usable margin for assignment."
            ),
            candidate_scores=candidates[:3],
            method="heuristic",
        )
    return ThermalMatchDecision(
        obs_id=observation.obs_id,
        matched_area_id="unmatched",
        confidence=max(0.2, min(0.58, 0.25 + max(score_gap, 0.0) * 10)),
        rationale=(
            f"The best deterministic score for {best.area_title} was {best.score:.2f}, "
            f"which was not clearly above the runner-up score of {second_score:.2f}."
        ),
        candidate_scores=candidates[:3],
        method="heuristic",
    )


def assign_thermal_observations(
    observations: list[ThermalObservation],
    areas: list[InspectionArea],
    photos: list[InspectionPhoto],
    reasoner: BaseReasoner | None = None,
) -> list[ThermalMatchDecision]:
    photo_lookup = {photo.photo_id: photo for photo in photos}
    area_lookup = {area.area_id: area for area in areas}
    decisions: list[ThermalMatchDecision] = []

    for observation in observations:
        candidates = score_area_candidates(observation, areas, photo_lookup)
        heuristic = _heuristic_decision(observation, candidates)
        decision = heuristic
        if reasoner and candidates[:3]:
            candidate_payload = []
            for candidate in candidates[:3]:
                area = area_lookup[candidate.area_id]
                supporting_images = [
                    photo_lookup[photo_id].image_path
                    for photo_id in candidate.supporting_photo_ids
                    if photo_id in photo_lookup
                ]
                candidate_payload.append(
                    {
                        "area_id": area.area_id,
                        "area_title": area.title,
                        "negative_description": area.negative_description,
                        "positive_description": area.positive_description,
                        "heuristic_score": candidate.score,
                        "supporting_images": supporting_images,
                    }
                )
            try:
                llm_decision = reasoner.resolve_thermal_match(observation, candidates[:3], candidate_payload)
                if llm_decision:
                    decision = llm_decision
            except Exception:
                decision = heuristic
        decisions.append(decision)
    return decisions


def group_decisions_by_area(decisions: list[ThermalMatchDecision]) -> dict[str, list[ThermalMatchDecision]]:
    grouped: dict[str, list[ThermalMatchDecision]] = defaultdict(list)
    for decision in decisions:
        grouped[decision.matched_area_id].append(decision)
    return dict(grouped)
