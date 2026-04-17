from __future__ import annotations

from collections import OrderedDict

from .llm import BaseReasoner, LLMGlobalNarrative, LLMAreaNarrative
from .models import (
    DDRReport,
    DDRAreaObservation,
    EvidenceBundle,
    IMAGE_NOT_AVAILABLE,
    InspectionDocument,
    MeasurementSummary,
    NOT_AVAILABLE,
    SeveritySummary,
    SourceIndexEntry,
    ThermalDocument,
    ThermalMatchDecision,
)


def build_evidence_bundles(
    inspection_doc: InspectionDocument,
    thermal_doc: ThermalDocument,
    decisions: list[ThermalMatchDecision],
) -> tuple[list[EvidenceBundle], list[ThermalMatchDecision]]:
    photo_lookup = {photo.photo_id: photo for photo in inspection_doc.photos}
    thermal_lookup = {observation.obs_id: observation for observation in thermal_doc.observations}
    decision_lookup: dict[str, list[ThermalMatchDecision]] = {}
    for decision in decisions:
        decision_lookup.setdefault(decision.matched_area_id, []).append(decision)

    bundles: list[EvidenceBundle] = []
    for area in inspection_doc.impacted_areas:
        area_photos = [photo_lookup[photo_id] for photo_id in area.photo_ids if photo_id in photo_lookup]
        missing_photo_ids = [photo_id for photo_id in area.photo_ids if photo_id not in photo_lookup]
        area_decisions = decision_lookup.get(area.area_id, [])
        area_thermal = [thermal_lookup[decision.obs_id] for decision in area_decisions if decision.obs_id in thermal_lookup]
        measurements = [
            MeasurementSummary(
                obs_id=observation.obs_id,
                date=observation.date,
                hotspot_c=observation.hotspot_c,
                coldspot_c=observation.coldspot_c,
                reflected_temp_c=observation.reflected_temp_c,
                emissivity=observation.emissivity,
            )
            for observation in area_thermal
        ]
        missing_items: list[str] = []
        if area.negative_description == NOT_AVAILABLE:
            missing_items.append("Negative-side description: Not Available")
        if area.positive_description == NOT_AVAILABLE:
            missing_items.append("Positive-side description: Not Available")
        for photo_id in missing_photo_ids:
            missing_items.append(f"{photo_id}: Image Not Available")
        if not area_photos:
            missing_items.append("Inspection images: Image Not Available")
        if not area_thermal:
            missing_items.append("Thermal evidence for this area: Not Available")

        conflicts: list[str] = []
        dates = sorted({observation.date for observation in area_thermal if observation.date != NOT_AVAILABLE})
        if len(dates) > 1:
            conflicts.append(f"Thermal observations span multiple dates: {', '.join(dates)}.")
        if any(decision.matched_area_id == "unmatched" for decision in area_decisions):
            conflicts.append("Some thermal observations could not be confidently tied to this area.")

        source_refs = [*area.source_refs]
        for photo in area_photos:
            source_refs.extend(photo.source_refs)
        for observation in area_thermal:
            source_refs.extend(observation.source_refs)

        bundles.append(
            EvidenceBundle(
                area_id=area.area_id,
                area_title=area.title,
                negative_description=area.negative_description,
                positive_description=area.positive_description,
                inspection_refs=[photo.photo_id for photo in area_photos],
                thermal_refs=[observation.obs_id for observation in area_thermal],
                inspection_photos=area_photos,
                thermal_observations=area_thermal,
                measurements=measurements,
                conflicts=conflicts,
                missing_items=missing_items,
                source_refs=source_refs,
            )
        )

    unmatched = [decision for decision in decisions if decision.matched_area_id == "unmatched"]
    return bundles, unmatched


def _keyword_flags(bundle: EvidenceBundle) -> set[str]:
    text = f"{bundle.negative_description} {bundle.positive_description}".lower()
    flags: set[str] = set()
    if any(term in text for term in ("damp", "seep", "leak", "efflorescence")):
        flags.add("moisture")
    if any(term in text for term in ("tile", "plumbing", "nahani", "bathroom", "wc")):
        flags.add("wet_area")
    if any(term in text for term in ("crack", "duct", "external wall")):
        flags.add("facade")
    if "ceiling" in text or "parking" in text:
        flags.add("spread")
    return flags


def _infer_root_cause(bundle: EvidenceBundle) -> str:
    flags = _keyword_flags(bundle)
    if {"moisture", "wet_area"} <= flags:
        return "Moisture ingress likely linked to wet-area tile joint failure, plumbing leakage, or poor sealing."
    if {"moisture", "facade"} <= flags:
        return "Moisture ingress is likely entering through cracks or weak waterproofing on the external wall or facade."
    if "moisture" in flags:
        return "Moisture ingress is the most likely cause, potentially driven by concealed leakage or failed waterproofing."
    if "facade" in flags:
        return "Facade or wall-surface distress appears to be the primary defect driver."
    return "Not Available"


def _infer_severity(bundle: EvidenceBundle) -> tuple[str, str]:
    score = 0
    text = f"{bundle.negative_description} {bundle.positive_description}".lower()
    if any(term in text for term in ("damp", "leak", "seepage", "efflorescence")):
        score += 2
    if any(term in text for term in ("crack", "plumbing", "hollowness")):
        score += 1
    if any(term in text for term in ("ceiling", "parking", "external wall")):
        score += 1
    deltas = [
        abs((measurement.hotspot_c or 0.0) - (measurement.coldspot_c or 0.0))
        for measurement in bundle.measurements
        if measurement.hotspot_c is not None and measurement.coldspot_c is not None
    ]
    if deltas and max(deltas) >= 4.0:
        score += 1

    if score >= 4:
        severity = "High"
    elif score >= 2:
        severity = "Moderate"
    else:
        severity = "Low"

    reasons: list[str] = []
    if any(term in text for term in ("damp", "leak", "seepage")):
        reasons.append("visible moisture-related distress is present")
    if any(term in text for term in ("crack", "external wall")):
        reasons.append("building envelope defects are referenced")
    if deltas:
        reasons.append(f"thermal spread reaches up to {max(deltas):.1f} C")
    if not reasons:
        reasons.append("available evidence is limited")
    return severity, "Severity is based on " + ", ".join(reasons) + "."


def _infer_actions(bundle: EvidenceBundle) -> list[str]:
    flags = _keyword_flags(bundle)
    actions: list[str] = []
    if "wet_area" in flags:
        actions.append("Inspect bathroom and plumbing fixtures, then repair leaking joints, traps, or concealed lines.")
        actions.append("Re-grout or reseal failed tile joints and replace hollow or loose tiles where needed.")
    if "facade" in flags:
        actions.append("Inspect external wall cracks and duct penetrations, then seal and waterproof the affected facade.")
    if "moisture" in flags:
        actions.append("Trace the moisture source before cosmetic repairs, then remove damaged plaster and repaint after drying.")
    if "spread" in flags:
        actions.append("Check adjacent upper-floor wet areas and ceiling interfaces to confirm the leakage path.")
    if not actions:
        actions.append("Perform a targeted site reinspection because the current evidence is not sufficient for a reliable repair scope.")

    deduped: list[str] = []
    seen: set[str] = set()
    for action in actions:
        if action not in seen:
            deduped.append(action)
            seen.add(action)
    return deduped


def _fallback_area_narrative(bundle: EvidenceBundle) -> LLMAreaNarrative:
    severity, severity_reasoning = _infer_severity(bundle)
    observation_parts = [bundle.negative_description]
    if bundle.positive_description != NOT_AVAILABLE:
        observation_parts.append(bundle.positive_description)
    if bundle.measurements:
        first = bundle.measurements[0]
        if first.hotspot_c is not None and first.coldspot_c is not None:
            observation_parts.append(
                f"Thermal evidence on {first.date} shows a hotspot of {first.hotspot_c:.1f} C and a coldspot of {first.coldspot_c:.1f} C."
            )
    return LLMAreaNarrative(
        observation_summary=" ".join(part for part in observation_parts if part and part != NOT_AVAILABLE),
        probable_root_cause=_infer_root_cause(bundle),
        severity=severity,
        severity_reasoning=severity_reasoning,
        recommended_actions=_infer_actions(bundle),
        additional_notes=bundle.conflicts,
        missing_or_unclear_information=bundle.missing_items,
    )


def _fallback_global_narrative(
    bundles: list[EvidenceBundle],
    unmatched: list[ThermalMatchDecision],
    thermal_doc: ThermalDocument,
) -> LLMGlobalNarrative:
    issue_counts = OrderedDict()
    for bundle in bundles:
        for flag in _keyword_flags(bundle):
            issue_counts[flag] = issue_counts.get(flag, 0) + 1

    dominant = []
    label_map = {
        "moisture": "moisture ingress, dampness, or leakage",
        "wet_area": "wet-area tile or plumbing defects",
        "facade": "external wall or facade distress",
        "spread": "secondary spread into ceilings or adjacent areas",
    }
    for flag, _ in sorted(issue_counts.items(), key=lambda item: item[1], reverse=True):
        dominant.append(label_map[flag])

    summary = [
        f"The inspection identifies {len(bundles)} impacted areas across the property.",
        (
            "The most consistent issue patterns are "
            + (", ".join(dominant[:3]) if dominant else "not clearly available from the source documents")
            + "."
        ),
        (
            f"The thermal report contains {len(thermal_doc.observations)} observations, "
            f"with {len(unmatched)} left unmatched where confidence was weak."
        ),
    ]
    root_causes = []
    for bundle in bundles:
        cause = _infer_root_cause(bundle)
        if cause != NOT_AVAILABLE and cause not in root_causes:
            root_causes.append(cause)

    recommended_actions = []
    for bundle in bundles:
        for action in _infer_actions(bundle):
            if action not in recommended_actions:
                recommended_actions.append(action)

    additional_notes = [
        "Only evidence visible in the provided inspection and thermal documents has been included in this report.",
        "Thermal observations that could not be tied to one area with enough confidence are shown separately instead of being forced into a conclusion.",
    ]
    if unmatched:
        additional_notes.append(f"{len(unmatched)} thermal observations remain unresolved and should be reviewed manually.")

    missing = []
    for bundle in bundles:
        for item in bundle.missing_items:
            if item not in missing:
                missing.append(f"{bundle.area_title}: {item}")

    return LLMGlobalNarrative(
        property_issue_summary=summary,
        probable_root_cause=root_causes or [NOT_AVAILABLE],
        recommended_actions=recommended_actions,
        additional_notes=additional_notes,
        missing_or_unclear_information=missing,
    )


def build_ddr_report(
    inspection_doc: InspectionDocument,
    thermal_doc: ThermalDocument,
    bundles: list[EvidenceBundle],
    unmatched: list[ThermalMatchDecision],
    reasoner: BaseReasoner | None = None,
) -> DDRReport:
    area_observations: list[DDRAreaObservation] = []
    severity_rows: list[SeveritySummary] = []
    llm_mode = reasoner.llm_mode if reasoner else "fallback"

    for bundle in bundles:
        try:
            narrative = reasoner.narrate_area_bundle(bundle) if reasoner else None
        except Exception:
            narrative = None
        if narrative is None:
            narrative = _fallback_area_narrative(bundle)

        severity = narrative.severity if narrative.severity in {"Low", "Moderate", "High"} else _infer_severity(bundle)[0]
        area_observations.append(
            DDRAreaObservation(
                area_id=bundle.area_id,
                area_title=bundle.area_title,
                observation_summary=narrative.observation_summary or NOT_AVAILABLE,
                probable_root_cause=narrative.probable_root_cause or NOT_AVAILABLE,
                severity=severity,
                severity_reasoning=narrative.severity_reasoning or NOT_AVAILABLE,
                recommended_actions=narrative.recommended_actions or [NOT_AVAILABLE],
                additional_notes=narrative.additional_notes or [],
                missing_or_unclear_information=narrative.missing_or_unclear_information or [],
                inspection_photo_paths=[photo.image_path for photo in bundle.inspection_photos] or [IMAGE_NOT_AVAILABLE],
                thermal_image_paths=[
                    observation.thermal_image_path
                    for observation in bundle.thermal_observations
                    if observation.thermal_image_path != IMAGE_NOT_AVAILABLE
                ]
                or [IMAGE_NOT_AVAILABLE],
                visible_image_paths=[
                    observation.visible_image_path
                    for observation in bundle.thermal_observations
                    if observation.visible_image_path != IMAGE_NOT_AVAILABLE
                ]
                or [IMAGE_NOT_AVAILABLE],
                thermal_measurements=bundle.measurements,
                evidence_refs=bundle.inspection_refs + bundle.thermal_refs,
            )
        )
        severity_rows.append(
            SeveritySummary(
                area_id=bundle.area_id,
                area_title=bundle.area_title,
                severity=severity,
                reasoning=narrative.severity_reasoning or NOT_AVAILABLE,
            )
        )

    try:
        global_narrative = reasoner.narrate_global_sections(bundles, unmatched, llm_mode) if reasoner else None
    except Exception:
        global_narrative = None
    if global_narrative is None:
        global_narrative = _fallback_global_narrative(bundles, unmatched, thermal_doc)

    source_index = [
        SourceIndexEntry(
            evidence_id=ref.evidence_id,
            document=ref.document,
            page_no=ref.page_no,
            kind=ref.kind,
            label=ref.label,
        )
        for ref in [*inspection_doc.source_index, *thermal_doc.source_index]
    ]
    return DDRReport(
        property_issue_summary=global_narrative.property_issue_summary,
        area_wise_observations=area_observations,
        probable_root_cause=global_narrative.probable_root_cause,
        severity_assessment=severity_rows,
        recommended_actions=global_narrative.recommended_actions,
        additional_notes=global_narrative.additional_notes,
        missing_or_unclear_information=global_narrative.missing_or_unclear_information,
        source_index=source_index,
        unmatched_thermal_observations=unmatched,
        llm_mode=llm_mode,
    )
