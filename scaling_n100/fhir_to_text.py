"""FHIR Bundle → clinical-note vignette text.

Companion to fhir_to_chr.py. For every "rich" Encounter (≥1 quantitative
Observation AND ≥1 Condition), renders a templated outpatient-visit
clinical note.

The text is structurally consistent (five paragraphs: header, measurement,
evaluation, treatment, care plan) but the surface vocabulary varies
because the underlying FHIR data is varied. This is the WebNLG-track
methodology of Text2KGBench: structured source data + templated rendering.

Output is paired with fhir_to_chr.py output by --start_index — same
encounter ordering, so vignette_001.txt corresponds to
vignette_001_gold_schema.ttl.

Usage::

    python scaling_n100/fhir_to_text.py \\
        --bundle <bundle.json> --vid_prefix vignette --start_index 1 \\
        --out_dir evaluation/corpus/vignettes/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fhir_to_chr import (  # type: ignore
    _attached_to_encounter, _build_resource_index, _label, _ref_id, is_rich,
)

_CODE_SYSTEM_LABEL = {
    "http://snomed.info/sct": "SNOMED",
    "http://loinc.org":       "LOINC",
    "https://loinc.org":      "LOINC",
}


def _inline_code(codeable_concept: dict) -> str:
    """Return ' (SYSTEM: code)' string for the most specific coding, or empty string."""
    for system, label in _CODE_SYSTEM_LABEL.items():
        for c in codeable_concept.get("coding", []):
            if c.get("system") == system and c.get("code"):
                return f" ({label}: {c['code']})"
    return ""


def _format_dt(dt: str) -> str:
    """ISO datetime → full ISO-8601 string (preserves seconds and timezone offset)."""
    if not dt:
        return "unspecified"
    return dt


def _patient_label(res: dict) -> str:
    n = res.get("name", [{}])[0]
    given = " ".join(n.get("given", []))
    family = n.get("family", "")
    full = f"{given} {family}".strip()
    return full or "the patient"


def _practitioner_label(res: dict | None, fallback_display: str) -> str:
    if not res:
        return fallback_display.strip() if fallback_display else "the attending physician"
    n = res.get("name", [{}])[0]
    family = n.get("family", "")
    return f"Dr. {family}".strip() or "the attending physician"


def render(bundle: dict, encounter: dict, res_idx: dict[str, dict]) -> str:
    """Render one rich encounter as a clinical-note vignette."""
    eid = encounter["id"]
    attached = _attached_to_encounter(bundle, eid)

    # Patient
    pat_ref = _ref_id(encounter.get("subject", {}).get("reference", ""))
    pat_res = res_idx.get(pat_ref) or {}
    pat_label = _patient_label(pat_res)
    pat_id = pat_ref[:8] if pat_ref else "unknown"

    # Provider
    prov_label = "the attending physician"
    for participant in encounter.get("participant", []):
        ind = participant.get("individual", {})
        pid = _ref_id(ind.get("reference", ""))
        prov_res = res_idx.get(pid)
        prov_label = _practitioner_label(prov_res, ind.get("display", ""))
        break

    # Care unit
    care_unit_label = "the outpatient clinic"
    sp = encounter.get("serviceProvider", {})
    sp_ref = _ref_id(sp.get("reference", ""))
    sp_res = res_idx.get(sp_ref)
    if sp_res and sp_res.get("name"):
        care_unit_label = sp_res["name"]
    elif sp.get("display"):
        care_unit_label = sp["display"]

    period = encounter.get("period", {})
    visit_date = _format_dt(period.get("start", ""))

    # Encounter type label
    enc_type = encounter.get("type", [{}])[0]
    enc_type_text = enc_type.get("text") or (
        enc_type.get("coding", [{}])[0].get("display", "Outpatient encounter")
    )

    # Header
    paragraphs = []
    paragraphs.append(
        f"CLINICAL VISIT NOTE — {care_unit_label}\n"
        f"{'=' * (len(care_unit_label) + 22)}\n\n"
        f"Patient: {pat_label} (ID: P-{pat_id})\n"
        f"Date of visit: {visit_date}\n"
        f"Care unit: {care_unit_label}\n"
        f"Attending physician: {prov_label}\n"
        f"Encounter type: {enc_type_text}"
    )

    # Measurement paragraph(s)
    measurements_text = []
    for obs in attached.get("Observation", []):
        vq = obs.get("valueQuantity")
        if not vq or "value" not in vq:
            continue
        obs_code = obs.get("code", {})
        code_text = obs_code.get("text") or (
            obs_code.get("coding", [{}])[0].get("display", "an observation")
        )
        code_inline = _inline_code(obs_code)
        unit = vq.get("unit", "")
        when = _format_dt(obs.get("effectiveDateTime", ""))
        measurements_text.append(
            f"On {when}, a {code_text}{code_inline} measurement was performed on "
            f"{pat_label}. The measurement result was {vq['value']} {unit}. "
            f"The measurement process was recorded with status \"completed\"."
        )
    if measurements_text:
        paragraphs.append("\n\n".join(measurements_text))

    # Condition / diagnostic paragraph(s)
    conds_text = []
    for cond in attached.get("Condition", []):
        code = cond.get("code", {})
        cond_label = code.get("text") or (
            code.get("coding", [{}])[0].get("display", "a clinical condition")
        )
        code_inline = _inline_code(code)
        when = _format_dt(cond.get("recordedDate", period.get("start", "")))
        sev = cond.get("severity", {})
        sev_label = (sev.get("text") or (sev.get("coding", [{}])[0].get("display", ""))) if sev else ""
        sev_sentence = f" Severity was assessed as \"{sev_label}\"." if sev_label else ""
        conds_text.append(
            f"{prov_label} evaluated the patient on {when}. The evaluation "
            f"produced a diagnostic statement of {cond_label}{code_inline}. The condition "
            f"was recorded on {when}.{sev_sentence}"
        )
    if conds_text:
        paragraphs.append("\n\n".join(conds_text))

    # Medication paragraph(s)
    meds_text = []
    for ma in attached.get("MedicationAdministration", []):
        med = ma.get("medicationCodeableConcept", {}) or {}
        med_text = med.get("text") or (
            med.get("coding", [{}])[0].get("display", "a medication")
        )
        when = _format_dt(ma.get("effectiveDateTime", period.get("start", "")))
        meds_text.append(
            f"{prov_label} initiated a treatment plan with {med_text}. The "
            f"first dose was administered on {when} by the clinical staff."
        )
    if meds_text:
        paragraphs.append("\n\n".join(meds_text))

    # Care plan paragraph (always emitted as a generic line so every
    # vignette hits all 5 expected sections; if Synthea didn't attach a
    # CarePlan we still produce a follow-up sentence to keep the corpus
    # structurally uniform).
    paragraphs.append(
        "A care plan was established that refers to follow-up monitoring "
        "and routine review consistent with the diagnosis."
    )

    return "\n\n".join(paragraphs) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--vid_prefix", required=True)
    ap.add_argument("--start_index", type=int, default=1)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    bundle = json.loads(args.bundle.read_text())
    res_idx = _build_resource_index(bundle)
    encounters = [e["resource"] for e in bundle.get("entry", []) if e["resource"].get("resourceType") == "Encounter"]
    rich = [e for e in encounters if is_rich(bundle, e)]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, enc in enumerate(rich):
        vid = f"{args.vid_prefix}_{args.start_index + i:03d}"
        text = render(bundle, enc, res_idx)
        out = args.out_dir / f"{vid}.txt"
        out.write_text(text)
    print(f"[OK] Wrote {len(rich)} text vignettes to {args.out_dir}")
    return len(rich)


if __name__ == "__main__":
    main()
