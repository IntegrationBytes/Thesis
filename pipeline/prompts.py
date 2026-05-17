"""Schema-agnostic prompt builders for the unified extraction pipeline.

Four prompt builders, all operating on a :class:`SchemaContext` that bundles
everything prompt construction needs (namespace, class list, property
inventory, shapes/tbox paths, track label):

    build_full_schema_prompt
        Best-case structural prompt — System A default on the schema track.

    build_ontology_prompt
        SULO n-ary Role-mediated pattern (CHR ontology track).

    build_correction_prompt
        SHACL-violation feedback for System B; schema-agnostic (takes
        ttl + violation text and asks the LLM to fix).

    build_self_correction_prompt
        System C's self-review prompt: same rules as System A's prompt
        plus the original graph, and a request to re-check without
        any SHACL output. Methodological contrast to System B.

The context is built via factory functions (``chr_context(track)``) rather
than a registry dict — adding a new schema is "write a factory" rather than
"edit a mapping."

Usage::

    from prompts import chr_context, build_full_schema_prompt
    ctx = chr_context("schema")
    prompt = build_full_schema_prompt(ctx, clinical_text)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
_PIPELINE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PIPELINE_DIR.parent
_CORPUS_ROOT = _PROJECT_ROOT / "evaluation" / "corpus"


# ---------------------------------------------------------------------------
# Schema context — a bundle of everything prompt construction needs.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaContext:
    """All information needed to build extraction + correction prompts.
    ``classes`` is the flat list of rdf:type values the LLM may emit.
    ``properties`` is a list of ``(name, domain, range)`` triples; ``range``
    may be either an ``xsd:`` datatype or one of the classes.
    The context also carries SHACL + TBox paths so downstream code (the System B validator, the evaluator) can pull them from one source."""
    name: str                           # "chr"
    track: str                          # "schema" | "ontology"
    namespace: str                      # primary domain URI
    prefix: str                         # domain prefix (e.g. "chr")
    tbox_path: Path
    shapes_path: Path
    classes: list[str] = field(default_factory=list)
    properties: list[tuple[str, str, str]] = field(default_factory=list)
    sulo_namespace: str = "https://w3id.org/sulo/"
    example_namespace: str = "http://example.org/clinical/"

    @property
    def prefix_block(self) -> str:
        """Turtle prefix declarations, ready to splice into a prompt."""
        return (
            f"  @prefix {self.prefix}:  <{self.namespace}> .\n"
            f"  @prefix ex:   <{self.example_namespace}> .\n"
            f"  @prefix sulo: <{self.sulo_namespace}> .\n"
            f"  @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .\n"
            f"  @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> ."
        )
# CHR schema — 25 classes, 20 properties
_CHR_NAMESPACE = (
    "https://w3id.org/shexmap/resource/ontology-schema/"
    "d285f599-dc2e-4bd0-83f3-df21defa8821/"
)

_CHR_CLASSES: list[str] = [
    "AnatomicalStructure", "CarePlan", "CareProviderRole", "CareUnit",
    "ClinicalCondition", "ClinicalVisit", "Device", "DiagnosticStatement",
    "EvaluationProcess", "InstrumentRole", "Measurement",
    "MeasurementProcess", "MedicalProcedure", "MedicationAdministration",
    "Occupation", "OutputRole", "PerformerRole", "Person",
    "PharmaceuticalDose", "PharmaceuticalDoseForm", "PharmaceuticalProduct",
    "ProcessStatus", "Severity", "SubjectOfCareRole", "TreatmentPlan",
]

# (property, domain, range). MedicalProcedure is the parent of
# MeasurementProcess, EvaluationProcess, and MedicationAdministration, so
# hasPerformer / hasPerformedDate / hasStatus apply to any of those.
_CHR_PROPERTIES: list[tuple[str, str, str]] = [
    ("hasCareProvider",          "ClinicalVisit",                      "Person"),
    ("hasCareUnit",              "ClinicalVisit",                      "CareUnit"),
    ("hasConditionEndDate",      "ClinicalCondition",                  "xsd:dateTime"),
    ("hasDevice",                "MeasurementProcess",                 "Device"),
    ("hasDoseForm",              "PharmaceuticalProduct",              "PharmaceuticalDoseForm"),
    ("hasDoseQuantity",          "PharmaceuticalProduct",              "PharmaceuticalDose"),
    ("hasLocatedIn",             "ClinicalCondition",                  "AnatomicalStructure"),
    ("hasMeasuredDate",          "Measurement",                        "xsd:dateTime"),
    ("hasMedicalProcedure",      "CarePlan",                           "MedicalProcedure"),
    ("hasObservation",           "EvaluationProcess",                  "DiagnosticStatement"),
    ("hasPatient",               "ClinicalVisit | MedicalProcedure",   "Person"),
    ("hasPerformedDate",         "MedicalProcedure",                   "xsd:dateTime"),
    ("hasPerformer",             "MedicalProcedure",                   "Person"),
    ("hasPharmaceuticalProduct", "MedicationAdministration",           "PharmaceuticalProduct"),
    ("hasQuantityValue",         "Measurement",                        "xsd:float"),
    ("hasRecordDate",            "ClinicalCondition",                  "xsd:dateTime"),
    ("hasResult",                "MeasurementProcess",                 "Measurement"),
    ("hasSeverity",              "ClinicalCondition",                  "Severity"),
    ("hasStatus",                "MedicalProcedure",                   "ProcessStatus"),
    ("hasUnit",                  "Measurement",                        "sulo:Unit"),
]


def chr_context(track: str) -> SchemaContext:
    """Build a :class:`SchemaContext` for the CHR schema on the given track.
    ``track="schema"`` uses the flat TBox and the schema-track SHACL.
    ``track="ontology"`` uses the OWL TBox and the ontology-track SHACL that
    enforces the SULO Role pattern.
    """
    if track not in ("schema", "ontology"):
        raise ValueError(
            f"track must be 'schema' or 'ontology', got {track!r}"
        )
    if track == "schema":
        tbox = _CORPUS_ROOT / "tbox" / "chr_schema.ttl"
        shapes = _CORPUS_ROOT / "shapes" / "chr_shacl_schema.ttl"
    else:
        tbox = _CORPUS_ROOT / "tbox" / "chr_ontology.owl.ttl"
        # RQ2 conformance target: role-targeted SHACL derived from the
        # supplied SULO ontology. The verbatim supplied file is preserved
        # as chr_shacl_ontology_supplied.ttl for traceability but is not
        # executed (the sequence-path constraints fire ~18 false positives
        # per well-formed graph — empirically unusable).
        shapes = _CORPUS_ROOT / "shapes" / "chr_shacl_ontology.ttl"
    return SchemaContext(
        name="chr",
        track=track,
        namespace=_CHR_NAMESPACE,
        prefix="chr",
        tbox_path=tbox,
        shapes_path=shapes,
        classes=list(_CHR_CLASSES),
        properties=list(_CHR_PROPERTIES),
    )


# ---------------------------------------------------------------------------
# Prompt builders.
# ---------------------------------------------------------------------------
def _class_lines(ctx: SchemaContext) -> str:
    return "\n".join(f"  - {ctx.prefix}:{c}" for c in ctx.classes)


def _property_lines(ctx: SchemaContext) -> str:
    return "\n".join(
        f"  - {ctx.prefix}:{p}  (domain: {d}, range: {r})"
        for p, d, r in ctx.properties
    )


def build_full_schema_prompt(ctx: SchemaContext, clinical_text: str) -> str:
    """Best-case structural prompt.
    Includes: prefix block, class list, property inventory with domain/range,
    six hard rules the SHACL validator will check, IRI-naming guidance, and
    the snippet. This is what System A runs by default, and also what
    Study 2 uses as the "upper-bound prompt" in the degradation ablation."""
    hard_rules = _full_hard_rules(ctx)
    return f"""You are an expert in Semantic Web technologies (RDF, OWL, SHACL)
working in the healthcare domain. Extract an RDF ABox from the clinical text
below, conforming to the Clinical Health Record ({ctx.prefix.upper()}) schema.

Use exactly these prefixes:
{ctx.prefix_block}

Allowed classes (use only these for rdf:type):
{_class_lines(ctx)}

Allowed properties (use them as direct binary predicates; respect the listed
domain and range):
{_property_lines(ctx)}

HARD RULES — the SHACL validator will reject output that violates these:
{hard_rules}

OUTPUT: Return ONLY valid Turtle — no prose, no code fences.

CLINICAL TEXT:
\"\"\"
{clinical_text.strip()}
\"\"\"
"""


def build_ontology_prompt(ctx: SchemaContext, clinical_text: str) -> str:
    """SULO n-ary Role-mediated prompt.
    Only meaningful on ``track="ontology"`` — raises if called on the flat schema track (the schema track's SHACL would reject the Role pattern).
    Walks through the role-to-entity mappings, temporal/quality/spatial attachment patterns, the MedicationAdministration direct-participation exception, and the six hard rules.
    """
    if ctx.track != "ontology":
        raise ValueError(
            "build_ontology_prompt only supports track='ontology'; "
            f"got track={ctx.track!r}"
        )
    return f"""You are an expert in Semantic Web technologies (RDF, OWL, SHACL) working in the healthcare domain. Extract an RDF ABox from the clinical text below using the SULO n-ary Role-mediated pattern.

Use exactly these prefixes:
{ctx.prefix_block}

Allowed {ctx.prefix.upper()} classes (use only these for rdf:type of domain nodes):
{_class_lines(ctx)}

ENCODING PATTERN — mandatory.

Instead of flat {ctx.prefix}: predicates, agent participation goes through *Role* individuals and SULO properties:
  Process  sulo:hasParticipant  Role .
  Role     sulo:isFeatureOf     Entity .

With these role-to-entity mappings (the SHACL Role shapes check them):
  {ctx.prefix}:SubjectOfCareRole  sulo:isFeatureOf -> {ctx.prefix}:Person
  {ctx.prefix}:CareProviderRole   sulo:isFeatureOf -> {ctx.prefix}:Person
  {ctx.prefix}:PerformerRole      sulo:isFeatureOf -> {ctx.prefix}:Person
  {ctx.prefix}:InstrumentRole     sulo:isFeatureOf -> {ctx.prefix}:Device
  {ctx.prefix}:OutputRole         sulo:isFeatureOf -> {ctx.prefix}:Measurement OR {ctx.prefix}:DiagnosticStatement

Temporal attachment:
  Process sulo:atTime TimeInstant . TimeInstant sulo:hasValue "..."^^xsd:dateTime .
  (TimeInstant may also be a sulo:EndTime for condition end dates.)

Quality / quantity attachment (Severity, ProcessStatus, DoseForm, Dose):
  Entity sulo:hasFeature Quality .

Spatial containment (CareUnit, AnatomicalStructure):
  Entity sulo:isIn Place .

Measurement value:
  Measurement sulo:hasValue  "152.0"^^xsd:float ;
              sulo:hasPart   <unit-iri-of-type-sulo:Unit> ;
              sulo:atTime    TimeInstant .

Care plan references:
  CarePlan sulo:refersTo MedicalProcedure .

DIRECT participation (only for MedicationAdministration -> PharmaceuticalProduct):
  MedicationAdministration sulo:hasParticipant PharmaceuticalProduct .
  (The drug is NOT wrapped in a Role. SubjectOfCareRole and PerformerRole
   around the same process ARE wrapped as Roles.)

HARD RULES — the SHACL validator will reject output that violates these:
  1. Every {ctx.prefix.upper()}-domain individual MUST have an rdf:type drawn from the
     allowed class list.
  2. Every Role individual MUST have sulo:isFeatureOf pointing at the
     correct target type (see role-to-entity mapping above).
  3. Every date MUST be attached via a TimeInstant carrying
     "...."^^xsd:dateTime in ISO-8601.
  4. Measurement's numeric value MUST be typed xsd:float.
  5. Use the SULO pattern strictly — do NOT emit flat {ctx.prefix}:hasPatient,
     {ctx.prefix}:hasPerformer, {ctx.prefix}:hasDevice, {ctx.prefix}:hasResult,
     {ctx.prefix}:hasRecordDate, etc. Those predicates are SCHEMA-track only;
     the ontology-track ABox uses SULO properties around Role reifications
     instead.
  6. Name individuals with descriptive ex: IRIs
     (e.g. ex:visit_2025_11_03, ex:role_VisitProvider_Rossi,
     ex:time_Eval_performed).

OUTPUT: Return ONLY valid Turtle — no prose, no code fences.

CLINICAL TEXT:
\"\"\"
{clinical_text.strip()}
\"\"\"
"""


def build_correction_prompt(
    ctx: SchemaContext, ttl_content: str, violations_text: str
) -> str:
    """System B's SHACL-feedback correction prompt.

    The prompt is schema-agnostic — it just shows the LLM the graph and
    the violation report and asks for a fix. ``ctx`` is passed so the
    prompt can remind the LLM which namespace/track the fix must live in;
    if the LLM strays, the next validation round catches it.
    """
    track_note = (
        "Use the flat {p}: predicates — not the SULO Role pattern."
        if ctx.track == "schema"
        else "Use the SULO n-ary Role-mediated pattern — not flat {p}: predicates."
    ).format(p=ctx.prefix)
    return f"""You generated the following RDF graph from a clinical snippet, but
it failed SHACL validation. Here are the violations:

{violations_text}

Please fix the graph so it conforms to the {ctx.prefix.upper()} schema ({ctx.track} track).
{track_note}

Return ONLY the corrected RDF Turtle. No markdown fences, no prose, no
explanations.

ORIGINAL GRAPH:
{ttl_content}
"""


def build_self_correction_prompt(
    ctx: SchemaContext, ttl_content: str, snippet_text: str
) -> str:
    """System C's self-review prompt — no SHACL, only the rules restated.

    System C is the unstructured-feedback control: it sees the same kind of
    rules System A's prompt had, plus the original graph, and is asked to
    re-check the graph against the rules without any SHACL output. This is
    the methodological contrast to System B (which does see SHACL violations).
    """
    class_list = ", ".join(f"{ctx.prefix}:{c}" for c in ctx.classes)
    prop_list = ", ".join(f"{ctx.prefix}:{p}" for p, _, _ in ctx.properties)
    track_rule = (
        f"Use flat {ctx.prefix}: predicates only — no SULO Role pattern."
        if ctx.track == "schema"
        else (
            f"Use the SULO n-ary Role-mediated pattern (Process "
            "sulo:hasParticipant Role; Role sulo:isFeatureOf Entity). "
            f"Do NOT emit flat {ctx.prefix}: predicates like hasPatient, "
            "hasPerformer, hasDevice."
        )
    )
    return f"""You are a Clinical Knowledge Engineer reviewing an RDF Turtle
graph for ontology compliance against the {ctx.prefix.upper()} schema
({ctx.track} track).

Review the graph below for errors against these rules, then return a
corrected version.

RULES:
1. Every individual must have an rdf:type drawn from the allowed class
   list: {class_list}.
2. Properties used must be drawn from the allowed list: {prop_list}
   (plus SULO properties on the ontology track).
3. {track_rule}
4. Dates must be typed xsd:dateTime in ISO-8601; Measurement values
   must be typed xsd:float.
5. Use ex: IRIs for individuals.

ORIGINAL CLINICAL SNIPPET:
{snippet_text}

GRAPH TO REVIEW AND CORRECT:
{ttl_content}

Return ONLY the corrected RDF Turtle. No markdown fences, no prose, no
explanations. If the graph is already correct, return it unchanged.
"""


# ---------------------------------------------------------------------------
# Shared "hard rules" block for the full-schema prompt, split out so the
# schema and ontology tracks can differ where it matters.
# ---------------------------------------------------------------------------
def _full_hard_rules(ctx: SchemaContext) -> str:
    p = ctx.prefix
    if ctx.track == "schema":
        return f"""  1. Every ABox individual MUST have an rdf:type drawn from the allowed class list.
  2. Subjects and objects of each property MUST match its declared domain/range.
     - {p}:hasCareProvider and {p}:hasPerformer point at a {p}:Person, never a unit/device.
     - {p}:hasDevice points at a {p}:Device, never a person.
     - {p}:hasSeverity points at a {p}:Severity individual, never a person.
     - {p}:hasObservation points at a {p}:DiagnosticStatement, never a person.
     - {p}:hasMedicalProcedure points at a {p}:MedicalProcedure (or a subclass
       thereof: MeasurementProcess, EvaluationProcess, MedicationAdministration).
  3. Every {p}:MeasurementProcess MUST have exactly one {p}:hasPatient and one
     {p}:hasResult.
  4. {p}:hasQuantityValue MUST be typed xsd:float (e.g. "152.0"^^xsd:float).
  5. Any *Date property MUST be typed xsd:dateTime in ISO-8601
     (e.g. "2025-11-03T09:45:00"^^xsd:dateTime).
  6. Name individuals with descriptive ex: IRIs
     (e.g. ex:patient_JohnAndersson, ex:visit_2025_11_03, ex:meas_BP_20251103)."""
    # ontology track falls back to build_ontology_prompt's inline rules
    # — build_full_schema_prompt on the ontology track is rare but supported.
    return f"""  1. Every {p.upper()}-domain individual MUST have an rdf:type drawn from the
     allowed class list.
  2. Agent participation MUST go through Role individuals:
     Process sulo:hasParticipant Role; Role sulo:isFeatureOf Entity.
  3. Every date MUST be attached via a TimeInstant carrying an
     "..."^^xsd:dateTime literal in ISO-8601.
  4. Measurement's numeric value MUST be typed xsd:float.
  5. Name individuals with descriptive ex: IRIs."""
