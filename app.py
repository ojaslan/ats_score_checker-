import streamlit as st
import pandas as pd

from ats.parser import extract_text
from ats.matcher import match_skills
from ats.scorer import ats_score

st.set_page_config(page_title="ATS Resume Checker", layout="wide")
st.title("📄 ATS Resume Checker")

# ===== Sidebar (Recruiter Inputs) =====
st.sidebar.header("🧑‍💼 Recruiter Job Details")

job_title = st.sidebar.text_input("Job Title")
job_desc = st.sidebar.text_area("Job Description")
skills_input = st.sidebar.text_input("Required Skills (comma-separated)")
min_exp = st.sidebar.number_input("Minimum Experience (Years)", 0, 20, 0)

skills = [s.strip().lower() for s in skills_input.split(",") if s.strip()]

st.sidebar.markdown("---")
st.sidebar.write("📌 Upload resumes below")

# ===== Resume Upload =====
uploaded_files = st.file_uploader(
    "Upload Candidate Resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

results = []

if uploaded_files and skills:
    for file in uploaded_files:
        resume_text = extract_text(file)

        matched, missing = match_skills(resume_text, skills)
        score = ats_score(matched, len(skills))

        results.append({
            "Candidate": file.name,
            "ATS Score (%)": score,
            "Matched Skills": ", ".join(matched),
            "Missing Skills": ", ".join(missing)
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="ATS Score (%)", ascending=False)

    st.subheader("📊 Candidate Ranking")
    st.dataframe(df, use_container_width=True)

    st.subheader("🏆 Top Candidate")
    st.success(f"{df.iloc[0]['Candidate']} — {df.iloc[0]['ATS Score (%)']}%")

else:
    st.info("👈 Enter job skills in sidebar and upload resumes")
