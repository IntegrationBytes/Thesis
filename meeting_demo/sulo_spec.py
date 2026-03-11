SULO_NAMESPACE = "https://w3id.org/sulo/"
SULO_TTL_URL = "https://w3id.org/sulo/sulo.ttl"

# ---------------------------------------------------------------------------
# SULO classes — taxonomy from paper Figure 2 (disjoint at each level)
#
# Classification decision flowchart (apply in order; first match wins):
#   1. Does it unfold in time?           → Process
#   2. Is it a physical entity?          → SpatialObject
#   3. Is it information about something? → InformationObject
#   4. Does it characterize something?  → Quality
#   5. Is it a context-dependent role?   → Role
#   6. Is it an intrinsic ability? (or default) → Capability
# ---------------------------------------------------------------------------
SULO_CLASSES = {
    # Top level (under owl:Thing)
    "Process": {
        "description": "Has temporal parts, unfolds in time, has duration, has objects as participants (e.g. heart transplant, administration of medication).",
    },
    "Object": {
        "description": "Maintains identity through time, does not have processes as parts, participates in processes.",
        "subclasses": ["SpatialObject", "Feature"],
    },
    "SpatialObject": {
        "parent": "Object",
        "description": "Exists in space (e.g. person, particle, paper document, hospital). Can gain/lose parts; features can change over time.",
    },
    "Feature": {
        "parent": "Object",
        "description": "Existentially depends on other things (objects or processes).",
        "subclasses": ["Role", "Quality", "Capability", "InformationObject"],
    },
    "Role": {
        "parent": "Feature",
        "description": "e.g. care subject, care provider, student, blood donor.",
    },
    "Quality": {
        "parent": "Feature",
        "description": "e.g. the redness of an apple.",
    },
    "Capability": {
        "parent": "Feature",
        "description": "e.g. capability to breathe or produce offspring.",
    },
    "InformationObject": {
        "parent": "Feature",
        "description": "Depends on and is about something (e.g. document content, guideline, formula).",
        "subclasses": ["Collection", "Quantity"],
    },
    "Collection": {
        "parent": "InformationObject",
        "description": "InformationObject pertaining to classes and mathematical sets.",
    },
    "Quantity": {
        "parent": "InformationObject",
        "description": "Captures a numerical aspect with a value and optional Unit.",
        "subclasses": ["Time", "Unit"],
    },
    "Time": {
        "parent": "Quantity",
        "description": "Measurement of time (SULO does not offer an ontology of time).",
        "subclasses": ["TimeInstant", "TimeInterval", "Duration"],
    },
    "Unit": {
        "parent": "Quantity",
        "description": "Unit of measure.",
    },
    "TimeInstant": {
        "parent": "Time",
        "description": "A point in time.",
    },
    "TimeInterval": {
        "parent": "Time",
        "description": "An interval of time.",
    },
    "Duration": {
        "parent": "Time",
        "description": "A duration of time.",
    },
    "StartTime": {
        "parent": "TimeInstant",
        "description": "Role-playing TimeInstant for start of Process or TimeInterval.",
    },
    "EndTime": {
        "parent": "TimeInstant",
        "description": "Role-playing TimeInstant for end of Process or TimeInterval.",
    },
}
SULO_OBJECT_PROPERTIES = {
    "hasParticipant": {"inverse": "isParticipantIn", "description": "Between Process and Object. When a Feature participates, the Object it isFeatureOf participates (role chain)."},
    "isPartOf": {"inverse": "hasPart", "description": "Transitive, reflexive: x is incorporated in y; removing x would change y's integrity or identity."},
    "isDirectPartOf": {"inverse": "hasDirectPart", "description": "Non-transitive subrelation of isPartOf for OWL cardinality constraints."},
    "isIn": {"inverse": "contains", "description": "Spatial, temporal, structural, or conceptual extent—where something is located."},
    "atTime": {"inverse": "isTimeOf", "description": "Between any entity and a time measurement—when something exists or occurs."},
    "precedes": {"description": "Between two processes: p1 ends before p2 begins (strict temporal ordering)."},
    "hasFeature": {"inverse": "isFeatureOf", "description": "Between any entity and a Feature (e.g. SpatialObject hasFeature Quality)."},
    "refersTo": {"inverse": "isReferredToIn", "description": "InformationObject refersTo what it mentions, describes, or represents."},
    "hasMember": {"description": "Between Collection and items (paper also hasItem / isItemIn for Collection)."},
}
SULO_DATA_PROPERTIES = {
    "hasValue": {"description": "Functional data property; single literal per InformationObject. SOLID pattern uses this exclusively for literals."},
}
# Flow Chart Implementation
SULO_CLASSIFICATION_FLOW = [
    ("Does it unfold in time?", "Process"),
    ("Is it a physical entity?", "SpatialObject"),
    ("Is it information about something?", "InformationObject"),
    ("Does it characterize something?", "Quality"),
    ("Is it a context-dependent role?", "Role"),
    ("Is it an intrinsic ability?", "Capability"),  # also default if none above match
]
PATTERN_SOLID = {
    "name": "SOLID (Single Object Literal Information Datum)",
    "idea": "Use SULO's single functional datatype property hasValue to assign a literal to an InformationObject. Push semantics out of data properties into instances (e.g. Temperature Quantity with hasValue and refersTo PATO).",
    "relations_used": ["hasValue", "hasFeature", "refersTo", "hasPart"],
}
PATTERN_PRO = {
    "name": "PRO (Process-Role-Object)",
    "idea": "Process hasParticipant (Role and isFeatureOf some Object). Role holders participate via inference: hasParticipant o isFeatureOf -> hasParticipant.",
    "relations_used": ["hasParticipant", "isFeatureOf"],
}
CLINICAL_TO_SULO_MAPPING = {
    "Clinical finding (finding)": "Process",
    "Procedure (procedure)": "Process",
    "Finding site (attribute)": "isIn",
    "Organism / Body structure / Environment or location": "SpatialObject",
    "Observable entity / Record artifact": "InformationObject",
    "Time (qualifier value)": "Time",
    "Unit of measure (qualifier value)": "Unit",
    "Role Group (attribute)": "hasFeature",
    "Using device / Causative agent / etc.": "hasParticipant",
}

# ---------------------------------------------------------------------------
# Our 20 clinical individuals — exact list and their SULO types per the paper
# Conditions/findings → Process; Sites → SpatialObject;
# Procedures → Process; Plans → Process with hasPart; Time → TimeInstant.
# Traceability: we use atTime (when) and link findings to the Process that
# produced them via hasPart (evidence process is part of the finding's
# assertion context). "Base of assertion" = which Process (e.g. biopsy)
# supports the finding: we encode as the diagnostic Process that hasPart
# or is associated with the finding. Paper does not define assertedDate/
# baseOfAssertion; we map them to atTime + hasPart/participant for
# compatibility with the paper's relations.
# ---------------------------------------------------------------------------
CLINICAL_INDIVIDUALS = [
    # (local_name, sulo_class, optional_comment)
    ("maria", "SpatialObject", "Example person; can be linked via Roles / hasParticipant from Processes."),
    ("site_RightBreast", "SpatialObject", "Example site / location used with isIn."),
    ("site_Liver", "SpatialObject", "Another example site / location."),
    ("cond_InvasiveDuctalCarcinoma", "Process", "Example finding modelled as a Process."),
    ("obs_PathologyFinding", "InformationObject", "Example observation as InformationObject with optional hasValue."),
    ("cond_CompleteRemission", "Process", "Example state / outcome modelled as a Process."),
    ("cond_MetastaticBreastCancer", "Process", "Example evolution of a prior finding, linked via hasPart or a custom relation."),
    ("proc_InitialBiopsy", "Process", "Example procedure modelled as a Process."),
    ("proc_PostTreatmentImaging", "Process", "Example imaging procedure."),
    ("proc_MetastaticImaging", "Process", "Example follow‑up imaging procedure."),
    ("proc_MetastaticBiopsy", "Process", "Example follow‑up biopsy procedure."),
    ("obs_AbdominalDiscomfort", "InformationObject", "Example symptom / observation as InformationObject."),
    ("proc_Lumpectomy", "Process", "Example treatment procedure."),
    ("proc_Radiotherapy", "Process", "Example treatment procedure."),
    ("proc_EndocrineTherapy", "Process", "Example treatment procedure."),
    ("proc_SystemicChemotherapy", "Process", "Example treatment procedure."),
    ("plan_CurativeTreatmentPlan", "Process", "Example plan as a Process with hasPart links to proc_* Processes."),
    ("plan_PalliativeTreatmentPlan", "Process", "Another example plan as a Process with hasPart links."),
    ("time_PathologyDate", "TimeInstant", "Example TimeInstant used via atTime."),
    ("time_MetastaticDiscoveryDate", "TimeInstant", "Another example TimeInstant used via atTime."),
]


def get_concept_list_for_prompt() -> str:
    """Format the 20 individuals as prompt lines: local name and SULO class from paper."""
    lines = []
    for name, sulo_class, comment in CLINICAL_INDIVIDUALS:
        line = f"  :{name}  a  sulo:{sulo_class}"
        if comment:
            line += f"   # {comment}"
        lines.append(line)
    return "\n".join(lines)


def get_predicate_rules_for_prompt() -> str:
    return """
FOR every Process that represents a finding / state (cond_*, obs_* as Process/InformationObject):
  • sulo:atTime → link to the relevant :time_* (TimeInstant) individual.
  • sulo:hasParticipant → :maria (Process hasParticipant Object).
  • For location: sulo:isIn → the relevant :site_* (SpatialObject).

FOR :cond_MetastaticBreastCancer (Process) or any similar “evolution of” Process:
  • Link to the primary finding: use sulo:hasPart or a dedicated relation for "evolution of" (paper does not define isEvolutionOf; hasPart to the primary Process or a custom relation are both acceptable modelling choices).

FOR plan_* (Process):
  • sulo:hasPart → list the constituent :proc_* Process individuals.
  • Intent (Curative/Palliative): use an InformationObject with sulo:hasValue "Curative"^^xsd:string and sulo:refersTo the plan (SOLID pattern) or a simple literal relation for readability.

FOR diagnostic proc_* (Process):
  • Link to the finding they support: sulo:hasPart (the finding Process as part of the diagnostic context) or inverse hasParticipant / refersTo as appropriate.

FOR TimeInstant individuals (:time_*):
  • Optionally sulo:hasValue with an xsd:date literal (SOLID) for the actual date; if unknown use placeholder "9999-01-01"^^xsd:date.
"""


def get_paper_overview_for_prompt() -> str:
    """Short overview"""
    return (
        "SULO (Simplified Upper-Level Ontology): 17 classes, 18 object properties (9+9 inverse), 1 data property (hasValue). "
        "Key classes: Process, Object → SpatialObject | Feature; Feature → Role, Quality, Capability, InformationObject; "
        "InformationObject → Collection, Quantity; Quantity → Time, Unit; Time → TimeInstant, TimeInterval, Duration. "
        "Key relations: hasParticipant (Process–Object), isPartOf/hasPart, isIn (location), atTime (when), hasFeature, refersTo, hasValue. "
        "SOLID: use hasValue for literals; PRO: Process hasParticipant (Role isFeatureOf Object). "
        "Findings / events → Process; Sites / locations → SpatialObject with isIn; Procedures → Process; Observables → InformationObject (paper Table 1 provides one example domain)."
    )
