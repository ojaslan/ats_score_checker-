import streamlit as st
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import pdfplumber

nltk.download("punkt")
nltk.download("stopwords")
STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z+ ]", " ", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def score_resume(resume_text, jd_text, skills):
    resume_tokens = clean_text(resume_text)
    jd_tokens = clean_text(jd_text)

    resume_counter = Counter(resume_tokens)

    jd_match = sum(1 for w in set(jd_tokens) if w in resume_counter)
    jd_score = (jd_match / max(len(set(jd_tokens)), 1)) * 60

    matched_skills = [s for s in skills if s.lower() in resume_text.lower()]
    skill_score = (len(matched_skills) / max(len(skills), 1)) * 40

    final_score = round(jd_score + skill_score, 2)
    return final_score, matched_skills

st.set_page_config(page_title="ATS Resume Scorer", layout="wide")

st.title("ATS Resume Scorer")
st.caption("HR defines JD and skills. Resume scored using NLP.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("HR Inputs")

    jd_text = st.text_area("Job Description", height=200)

    if "skills" not in st.session_state:
        st.session_state.skills = []

    skill_input = st.text_input("Add Skill")

    if st.button("Add Skill"):
        if skill_input and skill_input not in st.session_state.skills:
            st.session_state.skills.append(skill_input)

    if st.session_state.skills:
        for i, skill in enumerate(st.session_state.skills):
            c1, c2 = st.columns([4, 1])
            c1.write(skill)
            if c2.button("Remove", key=f"r{i}"):
                st.session_state.skills.pop(i)
                st.rerun()

with col2:
    st.subheader("Resume Upload")

    uploaded_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

    if uploaded_file and jd_text and st.session_state.skills:
        resume_text = extract_text_from_pdf(uploaded_file)
        score, matched_skills = score_resume(
            resume_text,
            jd_text,
            st.session_state.skills
        )

        st.metric("ATS Score", f"{score} / 100")
        st.write("Matched Skills:")
        st.write(", ".join(matched_skills) if matched_skills else "None")
    else:
        st.info("Add JD, skills and upload resume")