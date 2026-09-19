import re
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pint
from rapidfuzz import process, fuzz

ureg = pint.UnitRegistry()

REFERENCE_DATE_UTC = datetime(2026, 1, 1, tzinfo=timezone.utc)


@dataclass
class ProvenancePointer:
    source_id: str
    source_type: str
    extractor: str
    raw_text: str
    char_start: int
    char_end: int
    matched_text: str


@dataclass
class Claim:
    id: str
    case_id: str
    subject: str
    predicate: str
    value: float
    unit: Optional[str] = None
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    confidence: float = 1.0
    provenance: Optional[ProvenancePointer] = None
    subject_candidates: Optional[list] = None
    subject_resolution: Optional[str] = None


def make_utc(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_claim_interval(claim):
    start = make_utc(claim.time_start)
    end = make_utc(claim.time_end)
    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start
    if end < start:
        start, end = end, start
    return (start, end)


def intervals_overlap(i1, i2):
    s1, e1 = i1
    s2, e2 = i2
    return s1 <= e2 and s2 <= e1


def parse_time_from_text(text, reference_date=None):
    if reference_date is None:
        reference_date = REFERENCE_DATE_UTC
    reference_date = make_utc(reference_date)
    match = re.search(r"\bat\s+(\d{1,2})[:.](\d{2})\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return reference_date.replace(hour=hour, minute=minute, second=0, microsecond=0)


ENTITY_ALIASES = {
    "amoxicillin": ["amoxicillin", "amoxil", "amox"],
    "ibuprofen": ["ibuprofen", "advil", "motrin"],
    "paracetamol": ["paracetamol", "acetaminophen", "tylenol", "panadol"],
    "aspirin": ["aspirin", "asa", "acetylsalicylic acid"],
    "patient": ["patient", "pt"],
}

ENTITY_TYPE = {
    "amoxicillin": "drug",
    "ibuprofen": "drug",
    "paracetamol": "drug",
    "aspirin": "drug",
    "patient": "person",
}

ALIAS_TO_CANONICAL = {}
for canonical, aliases in ENTITY_ALIASES.items():
    for alias in aliases:
        ALIAS_TO_CANONICAL[alias.lower()] = canonical

ALL_ALIASES = list(ALIAS_TO_CANONICAL.keys())

PREDICATE_PATTERNS = [
    ("temperature", r"\b(temperature|temp|fever)\b"),
    ("weight", r"\b(weight|weighs|weighed)\b"),
    ("dose", r"\b(dose|dosed|received|administered|given|took)\b"),
]

PREDICATE_SUBJECT_TYPE = {
    "dose": "drug",
    "temperature": "person",
    "weight": "person",
}

VALUE_UNIT_PATTERN = r"(\d+\.?\d*)\s*(°C|°F|C|F|mg|kg|g|ml|L|lbs|lb)\b"

UNIT_CANON = {
    "C": "degC",
    "°C": "degC",
    "F": "degF",
    "°F": "degF",
    "lbs": "lb",
}


def detect_predicate(text):
    text_lower = text.lower()
    for name, pattern in PREDICATE_PATTERNS:
        if re.search(pattern, text_lower):
            return name
    return "measurement"


def find_subject_candidates(text, subject_type=None):
    text_lower = text.lower()
    candidates = {}

    for alias in sorted(ALL_ALIASES, key=len, reverse=True):
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, text_lower):
            canonical = ALIAS_TO_CANONICAL[alias]
            if subject_type is not None and ENTITY_TYPE.get(canonical) != subject_type:
                continue
            current = candidates.get(canonical)
            if current is None or current["score"] < 100:
                candidates[canonical] = {
                    "canonical": canonical,
                    "alias": alias,
                    "score": 100,
                    "method": "exact",
                }

    words = re.findall(r"[a-z]{4,}", text_lower)
    for word in words:
        matches = process.extract(
            word, ALL_ALIASES, scorer=fuzz.ratio, score_cutoff=85, limit=5
        )
        for alias, score, _ in matches:
            canonical = ALIAS_TO_CANONICAL[alias]
            if subject_type is not None and ENTITY_TYPE.get(canonical) != subject_type:
                continue
            current = candidates.get(canonical)
            if current is None or current["score"] < score:
                candidates[canonical] = {
                    "canonical": canonical,
                    "alias": alias,
                    "score": int(score),
                    "method": "fuzzy",
                }

    return sorted(candidates.values(), key=lambda c: c["score"], reverse=True)


def resolve_subject_with_candidates(text, subject_type=None):
    candidates = find_subject_candidates(text, subject_type=subject_type)
    if not candidates:
        return None, [], "unresolved"
    if len(candidates) == 1:
        return candidates[0]["canonical"], candidates, "resolved"
    return None, candidates, "ambiguous"


def check_conflict(claim1, claim2):
    if claim1.case_id != claim2.case_id:
        return "No conflict: different cases."
    if claim1.subject != claim2.subject or claim1.predicate != claim2.predicate:
        return "No conflict: different subjects or predicates."

    interval1 = get_claim_interval(claim1)
    interval2 = get_claim_interval(claim2)
    if interval1 is None or interval2 is None:
        return "No conflict confirmed: missing time information."
    if not intervals_overlap(interval1, interval2):
        return "No conflict: times do not overlap."

    try:
        q1 = ureg.Quantity(claim1.value, claim1.unit)
        q2 = ureg.Quantity(claim2.value, claim2.unit)
        base1 = q1.to_base_units()
        if base1.check("[temperature]"):
            canon = "degC"
        elif base1.check("[mass]"):
            canon = "gram"
        elif base1.check("[volume]"):
            canon = "mL"
        else:
            canon = None
        if canon is not None:
            v1 = q1.to(canon)
            v2 = q2.to(canon)
        else:
            v1 = base1
            v2 = q2.to_base_units()
    except Exception as e:
        return f"Could not compare units: {e}"

    relative_diff = abs(v1.magnitude - v2.magnitude) / max(
        abs(v1.magnitude), abs(v2.magnitude), 1e-9
    )
    if relative_diff > 0.01:
        return f"CONFLICT DETECTED: {claim1.value} {claim1.unit} vs {claim2.value} {claim2.unit}"
    return "No conflict: values are equivalent."


def extract_claim_from_text(text, case_id, claim_id, source_id=None, reference_date=None):
    if source_id is None:
        source_id = f"{case_id}:{claim_id}"
    match = re.search(VALUE_UNIT_PATTERN, text)
    if not match:
        return None
    value, unit_raw = match.groups()
    unit = UNIT_CANON.get(unit_raw, unit_raw)
    predicate = detect_predicate(text)
    subject_type = PREDICATE_SUBJECT_TYPE.get(predicate)
    subject, candidates, status = resolve_subject_with_candidates(text, subject_type=subject_type)
    if status == "unresolved":
        return None
    subject_value = subject if status == "resolved" else "UNRESOLVED"
    provenance = ProvenancePointer(
        source_id=source_id,
        source_type="text",
        extractor="general_extractor_v2",
        raw_text=text,
        char_start=match.start(),
        char_end=match.end(),
        matched_text=text[match.start():match.end()],
    )
    time_start = parse_time_from_text(text, reference_date=reference_date)
    return Claim(
        id=claim_id,
        case_id=case_id,
        subject=subject_value,
        predicate=predicate,
        value=float(value),
        unit=unit,
        time_start=time_start,
        confidence=0.9,
        provenance=provenance,
        subject_candidates=candidates,
        subject_resolution=status,
    )


DB_PATH = "umer_demo.db"


def init_db(path=DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS claims (
            case_id TEXT NOT NULL,
            id TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT,
            time_start TEXT,
            time_end TEXT,
            confidence REAL,
            source_id TEXT,
            source_type TEXT,
            extractor TEXT,
            raw_text TEXT,
            char_start INTEGER,
            char_end INTEGER,
            matched_text TEXT,
            subject_resolution TEXT,
            subject_candidates TEXT,
            PRIMARY KEY (case_id, id)
        );
        CREATE TABLE IF NOT EXISTS conflicts (
            conflict_id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            claim_1_id TEXT NOT NULL,
            claim_2_id TEXT NOT NULL,
            message TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            UNIQUE(case_id, claim_1_id, claim_2_id)
        );
        """
    )
    conn.commit()
    conn.close()


def dt_to_iso(dt):
    if dt is None:
        return None
    return dt.isoformat()


def iso_to_dt(value):
    if value is None:
        return None
    return datetime.fromisoformat(value)


def save_claims_to_sqlite(claims, path=DB_PATH):
    init_db(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    for claim in claims:
        p = claim.provenance
        cur.execute(
            """
            INSERT OR REPLACE INTO claims (
                case_id, id, subject, predicate, value, unit,
                time_start, time_end, confidence,
                source_id, source_type, extractor, raw_text,
                char_start, char_end, matched_text,
                subject_resolution, subject_candidates
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                claim.case_id, claim.id, claim.subject, claim.predicate,
                claim.value, claim.unit,
                dt_to_iso(claim.time_start), dt_to_iso(claim.time_end),
                claim.confidence,
                p.source_id if p else None,
                p.source_type if p else None,
                p.extractor if p else None,
                p.raw_text if p else None,
                p.char_start if p else None,
                p.char_end if p else None,
                p.matched_text if p else None,
                claim.subject_resolution,
                json.dumps(claim.subject_candidates if claim.subject_candidates else []),
            ),
        )
    conn.commit()
    conn.close()


def load_claims_from_sqlite(path=DB_PATH):
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM claims ORDER BY case_id, id")
    rows = cur.fetchall()
    conn.close()
    claims = []
    for row in rows:
        provenance = None
        if row["source_id"] is not None:
            provenance = ProvenancePointer(
                source_id=row["source_id"],
                source_type=row["source_type"],
                extractor=row["extractor"],
                raw_text=row["raw_text"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                matched_text=row["matched_text"],
            )
        candidates = json.loads(row["subject_candidates"]) if row["subject_candidates"] else []
        claims.append(
            Claim(
                id=row["id"],
                case_id=row["case_id"],
                subject=row["subject"],
                predicate=row["predicate"],
                value=row["value"],
                unit=row["unit"],
                time_start=iso_to_dt(row["time_start"]),
                time_end=iso_to_dt(row["time_end"]),
                confidence=row["confidence"],
                provenance=provenance,
                subject_candidates=candidates,
                subject_resolution=row["subject_resolution"],
            )
        )
    return claims


def find_and_save_conflicts(claims, path=DB_PATH):
    init_db(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    count = 0
    now = datetime.now(timezone.utc).isoformat()
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            result = check_conflict(claims[i], claims[j])
            if result.startswith("CONFLICT DETECTED"):
                cur.execute(
                    """
                    INSERT OR REPLACE INTO conflicts
                        (case_id, claim_1_id, claim_2_id, message, detected_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (claims[i].case_id, claims[i].id, claims[j].id, result, now),
                )
                count += 1
    conn.commit()
    conn.close()
    return count


def load_conflicts_from_sqlite(path=DB_PATH):
    init_db(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM conflicts ORDER BY conflict_id")
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def seed_demo_data(path=DB_PATH):
    """Fill the demo database once with multimodal example evidence."""
    init_db(path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM claims")
    if cur.fetchone()[0] > 0:
        conn.close()
        return

    T10 = "2026-01-01T10:00:00+00:00"
    T1005 = "2026-01-01T10:05:00+00:00"
    T1003 = "2026-01-01T10:03:00+00:00"
    T1008 = "2026-01-01T10:08:00+00:00"

    SEED = [
        dict(id="text_0", subject="amoxicillin", predicate="dose", value=500.0, unit="mg",
             ts=T10, te=None, conf=0.9, source_id="demo:text_0", stype="text",
             extractor="general_extractor_v2",
             raw="Patient received 500 mg amoxicillin at 10:00", matched="500 mg",
             res="resolved",
             cands='[{"canonical": "amoxicillin", "alias": "amoxicillin", "score": 100, "method": "exact"}]'),
        dict(id="csv_2", subject="amoxicillin", predicate="dose", value=0.5, unit="g",
             ts=T10, te=None, conf=1.0, source_id="lab_results.csv:row=2:col=value", stype="table",
             extractor="csv_adapter_v1",
             raw="2026-01-01T10:00:00+00:00,amoxicillin,dose,0.5,g,1.0", matched="0.5",
             res="resolved",
             cands='[{"canonical": "amoxicillin", "alias": "amoxicillin", "score": 100, "method": "table_direct"}]'),
        dict(id="csv_3", subject="amoxicillin", predicate="dose", value=5.0, unit="g",
             ts=T10, te=None, conf=1.0, source_id="lab_results.csv:row=3:col=value", stype="table",
             extractor="csv_adapter_v1",
             raw="2026-01-01T10:00:00+00:00,amoxicillin,dose,5,g,1.0", matched="5",
             res="resolved",
             cands='[{"canonical": "amoxicillin", "alias": "amoxicillin", "score": 100, "method": "table_direct"}]'),
        dict(id="img_0", subject="amoxicillin", predicate="dose", value=600.0, unit="mg",
             ts=T10, te=None, conf=0.63, source_id="medical_note.png", stype="image",
             extractor="easyocr_v1 (AI)",
             raw="Patient received 600 mg amoxicillin at 10.00", matched="600 mg",
             res="resolved",
             cands='[{"canonical": "amoxicillin", "alias": "amoxicillin", "score": 100, "method": "exact"}]'),
        dict(id="text_1", subject="patient", predicate="temperature", value=38.5, unit="degC",
             ts=T1005, te=None, conf=0.9, source_id="demo:text_1", stype="text",
             extractor="general_extractor_v2",
             raw="Patient temperature was 38.5 C at 10:05", matched="38.5 C",
             res="resolved",
             cands='[{"canonical": "patient", "alias": "patient", "score": 100, "method": "exact"}]'),
        dict(id="sensor_3", subject="patient", predicate="temperature", value=39.9, unit="degC",
             ts=T1003, te=T1008, conf=1.0, source_id="sensor_log.csv:row=3:sensor=therm_b", stype="sensor",
             extractor="sensor_adapter_v1",
             raw="2026-01-01T10:03:00+00:00,2026-01-01T10:08:00+00:00,therm_b,temperature,39.9,degC",
             matched="39.9", res="resolved",
             cands='[{"canonical": "patient", "alias": "therm_b", "score": 100, "method": "sensor_direct"}]'),
        dict(id="text_2", subject="UNRESOLVED", predicate="dose", value=500.0, unit="mg",
             ts=T10, te=None, conf=0.9, source_id="demo:text_2", stype="text",
             extractor="general_extractor_v2",
             raw="Patient received 500 mg aspirin or ibuprofen at 10:00", matched="500 mg",
             res="ambiguous",
             cands='[{"canonical": "aspirin", "alias": "aspirin", "score": 100, "method": "exact"}, {"canonical": "ibuprofen", "alias": "ibuprofen", "score": 100, "method": "exact"}]'),
    ]

    for s in SEED:
        cs = s["raw"].find(s["matched"])
        if cs < 0:
            cs = 0
        ce = cs + len(s["matched"])
        cur.execute(
            """
            INSERT OR REPLACE INTO claims (
                case_id, id, subject, predicate, value, unit,
                time_start, time_end, confidence,
                source_id, source_type, extractor, raw_text,
                char_start, char_end, matched_text,
                subject_resolution, subject_candidates
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "case_demo", s["id"], s["subject"], s["predicate"], s["value"], s["unit"],
                s["ts"], s["te"], s["conf"], s["source_id"], s["stype"], s["extractor"],
                s["raw"], cs, ce, s["matched"], s["res"], s["cands"],
            ),
        )
    conn.commit()
    conn.close()

    claims = load_claims_from_sqlite(path)
    find_and_save_conflicts(claims, path)
