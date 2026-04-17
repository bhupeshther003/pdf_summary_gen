from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import streamlit as st

from ddr_report.config import Settings, save_provider_settings_to_dotenv
from ddr_report.pipeline import run_pipeline


APP_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = APP_ROOT / "sample_outputs" / "streamlit_runs"


def _save_uploaded_pdf(uploaded_file, target_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    target_path = target_dir / f"{Path(uploaded_file.name).stem}{suffix}"
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _existing_image_paths(paths: list[str], limit: int = 6) -> list[str]:
    result: list[str] = []
    for path in paths:
        if path == "Image Not Available":
            continue
        file_path = Path(path)
        if file_path.exists():
            result.append(str(file_path))
        if len(result) >= limit:
            break
    return result


def _load_report_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


st.set_page_config(page_title="DDR Report Generator", page_icon=":bar_chart:", layout="wide")

st.title("DDR Report Generator")
st.caption("Upload an inspection report and a thermal report, then generate a structured DDR report with JSON and PDF outputs.")

with st.sidebar:
    st.header("Configuration")
    base_settings = Settings.from_env()
    provider = "gemini"
    default_api_key = base_settings.gemini_api_key
    default_model = base_settings.gemini_model
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=default_api_key or "",
        help="If provided, the app can use Gemini for thermal matching and narrative generation.",
    )
    save_key = st.checkbox("Save key to local .env", value=False)
    use_llm = st.checkbox("Enable LLM reasoning", value=bool(default_api_key))
    model_name = st.text_input("Model", value=default_model)
    enable_ocr = st.checkbox("Enable OCR fallback", value=base_settings.enable_ocr)
    st.markdown(
        "Gemini is the only supported provider. The app reads provider configuration from system environment variables, `.env`, or Streamlit secrets. "
        "Without a Gemini key, it still runs in deterministic fallback mode."
    )

left, right = st.columns(2)
with left:
    inspection_file = st.file_uploader("Inspection Report PDF", type=["pdf"], accept_multiple_files=False)
with right:
    thermal_file = st.file_uploader("Thermal Report PDF", type=["pdf"], accept_multiple_files=False)

generate = st.button("Generate DDR Report", type="primary", use_container_width=True)

if generate:
    if inspection_file is None or thermal_file is None:
        st.error("Please upload both the inspection PDF and the thermal PDF.")
    else:
        settings = replace(Settings.from_env(), llm_provider=provider)
        if model_name:
            settings = replace(settings, gemini_model=model_name)
        if api_key_input:
            settings = replace(settings, gemini_api_key=api_key_input)
            if save_key:
                env_path = save_provider_settings_to_dotenv(provider, api_key_input, model=settings.selected_model)
                st.sidebar.success(f"Saved {provider.upper()} configuration to {env_path}")
        if enable_ocr:
            settings = replace(settings, enable_ocr=True)
        settings = replace(settings, llm_enabled=bool(settings.selected_api_key))
        if not use_llm:
            settings = replace(settings, llm_enabled=False)

        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ddr-ui-", dir=str(OUTPUT_ROOT)) as temp_dir:
            temp_path = Path(temp_dir)
            inspection_path = _save_uploaded_pdf(inspection_file, temp_path)
            thermal_path = _save_uploaded_pdf(thermal_file, temp_path)
            with st.spinner("Generating DDR report..."):
                result = run_pipeline(
                    inspection_path=inspection_path,
                    thermal_path=thermal_path,
                    out_dir=OUTPUT_ROOT,
                    settings=settings,
                )

        report = _load_report_json(result.report_json_path)
        st.success("DDR report generated successfully.")
        st.code(result.run_dir, language="text")

        metric_1, metric_2, metric_3, metric_4 = st.columns(4)
        metric_1.metric("Impacted Areas", len(report["area_wise_observations"]))
        metric_2.metric(
            "Matched Thermal",
            sum(1 for area in report["area_wise_observations"] for path in area["thermal_image_paths"] if path != "Image Not Available"),
        )
        metric_3.metric("Unmatched Thermal", len(report["unmatched_thermal_observations"]))
        metric_4.metric("LLM Mode", report["llm_mode"])

        pdf_bytes = Path(result.report_pdf_path).read_bytes()
        json_bytes = Path(result.report_json_path).read_bytes()
        html_bytes = Path(result.report_html_path).read_bytes()
        download_1, download_2, download_3 = st.columns(3)
        download_1.download_button("Download PDF", pdf_bytes, file_name=Path(result.report_pdf_path).name, use_container_width=True)
        download_2.download_button("Download JSON", json_bytes, file_name=Path(result.report_json_path).name, use_container_width=True)
        download_3.download_button("Download HTML", html_bytes, file_name=Path(result.report_html_path).name, use_container_width=True)

        tab_summary, tab_areas, tab_unmatched = st.tabs(["Summary", "Areas", "Unmatched Thermal"])

        with tab_summary:
            st.subheader("Property Issue Summary")
            for item in report["property_issue_summary"]:
                st.write(f"- {item}")

            st.subheader("Probable Root Cause")
            for item in report["probable_root_cause"]:
                st.write(f"- {item}")

            st.subheader("Recommended Actions")
            for item in report["recommended_actions"]:
                st.write(f"- {item}")

            st.subheader("Missing Or Unclear Information")
            for item in report["missing_or_unclear_information"]:
                st.write(f"- {item}")

        with tab_areas:
            for area in report["area_wise_observations"]:
                with st.expander(f"{area['area_title']} | {area['severity']}", expanded=False):
                    st.write(area["observation_summary"])
                    st.write(f"Probable Root Cause: {area['probable_root_cause']}")
                    st.write(f"Severity Reasoning: {area['severity_reasoning']}")

                    if area["recommended_actions"]:
                        st.write("Recommended Actions")
                        for item in area["recommended_actions"]:
                            st.write(f"- {item}")

                    if area["additional_notes"]:
                        st.write("Additional Notes")
                        for item in area["additional_notes"]:
                            st.write(f"- {item}")

                    if area["missing_or_unclear_information"]:
                        st.write("Missing Or Unclear Information")
                        for item in area["missing_or_unclear_information"]:
                            st.write(f"- {item}")

                    inspection_images = _existing_image_paths(area["inspection_photo_paths"], limit=6)
                    thermal_images = _existing_image_paths(area["thermal_image_paths"], limit=3)
                    visible_images = _existing_image_paths(area["visible_image_paths"], limit=3)

                    if inspection_images:
                        st.write("Inspection Images")
                        st.image(inspection_images, width=220)
                    if thermal_images:
                        st.write("Thermal Images")
                        st.image(thermal_images, width=220)
                    if visible_images:
                        st.write("Visible Companion Images")
                        st.image(visible_images, width=220)

                    if area["thermal_measurements"]:
                        st.write("Thermal Readings")
                        st.dataframe(area["thermal_measurements"], use_container_width=True, hide_index=True)

        with tab_unmatched:
            if not report["unmatched_thermal_observations"]:
                st.info("No unmatched thermal observations.")
            else:
                for item in report["unmatched_thermal_observations"]:
                    st.write(f"**{item['obs_id']}**")
                    st.write(item["rationale"])
                    st.write(f"Confidence: {item['confidence']:.2f}")
                    st.divider()
