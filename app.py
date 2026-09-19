import json

import pandas as pd
import streamlit as st

import engine

st.set_page_config(page_title="UMER Evidence Inspector", layout="wide")
st.title("🔍 UMER Evidence Inspector")
st.markdown("Multimodal evidence with provenance, temporal alignment, and conflict detection.")

engine.init_db()
engine.seed_demo_data()

st.sidebar.header("➕ Add new evidence")
new_sentence = st.sidebar.text_area(
    "Paste a sentence (dose, temperature, weight):",
    "Patient received 500 mg amoxicillin at 10:00",
)

if st.sidebar.button("Extract claim"):
    existing = engine.load_claims_from_sqlite()
    claim = engine.extract_claim_from_text(
        new_sentence,
        case_id="case_demo",
        claim_id=f"user_{len(existing) + 1}",
        source_id="user_input",
    )
    if claim is None:
        st.sidebar.warning("No claim found. Need a known subject and a value+unit.")
    else:
        engine.save_claims_to_sqlite([claim])
        engine.find_and_save_conflicts(engine.load_claims_from_sqlite())
        st.sidebar.success(
            f"Saved {claim.id}: {claim.subject} {claim.predicate} = {claim.value} {claim.unit}"
        )
        st.rerun()


def md_table(df, columns):
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df[columns].iterrows():
        cells = []
        for c in columns:
            value = row[c]
            if value is None:
                value = ""
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return chr(10).join(lines)


claims = engine.load_claims_from_sqlite()
conflicts = engine.load_conflicts_from_sqlite()

st.header("📄 Extracted Claims")
if claims:
    claims_df = pd.DataFrame([c.__dict__ for c in claims])
    claims_df["time_start"] = claims_df["time_start"].apply(lambda d: d.isoformat() if d else "")
    st.markdown(md_table(claims_df, ["id", "subject", "predicate", "value", "unit", "time_start", "source_type_placeholder"] if False else ["id", "subject", "predicate", "value", "unit", "time_start", "subject_resolution"]))
else:
    st.info("No claims yet.")

st.header("⚠️ Detected Conflicts")
if conflicts:
    st.markdown(md_table(pd.DataFrame(conflicts), ["claim_1_id", "claim_2_id", "message"]))
else:
    st.success("No conflicts detected.")

st.header("🔎 Provenance & Ambiguity Deep Dive")
for c in claims:
    title = f"Claim {c.id}: {c.subject} {c.predicate} = {c.value} {c.unit} [{c.provenance.source_type if c.provenance else '?'}]"
    with st.expander(title):
        st.write(f"**Source:** {c.provenance.source_id if c.provenance else 'none'}")
        st.write(f"**Extractor:** {c.provenance.extractor if c.provenance else 'none'}")
        st.write(f"**Confidence:** {c.confidence}")
        st.write(f"**Resolution:** {c.subject_resolution}")
        if c.subject_candidates:
            st.write("**Candidates:**", c.subject_candidates)
        st.write("**Original source text:**")
        st.code(c.provenance.raw_text if c.provenance else "", language="text")
        st.write(f"**Matched:** '{c.provenance.matched_text if c.provenance else ''}' (chars {c.provenance.char_start if c.provenance else ''}:{c.provenance.char_end if c.provenance else ''})")
