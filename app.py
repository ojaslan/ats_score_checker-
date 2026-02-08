import streamlit as st
import fitz
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- NLTK CLOUD FIX ----------
NLTK_PATH = "/tmp/nltk_data"
if not os.path.exists(NLTK_PATH):
    os.makedirs(NLTK_PATH)

nltk.data.path.append(NLTK_PATH)

for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_PATH)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------- STREAMLIT CONFIG ----------
st.set_page_config("Cloud ATS", layout="wide")
st.title("☁️ Intelligent ATS – Cloud Edition (NLTK)")

# ---------- PDF EXTRACT ----------
def extract_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text.lower()

# ---------- PREPROCESS ----------
def preprocess(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)

# ---------- EXPERIENCE ----------
def extract_experience(text):
    matches = re.findall(r"(\d+\.?\d*)\s+year", text)
    return max([float(x) for x in matches], default=0)

# ---------- SKILL MATCH ----------
def skill_match(text, skills):
    return [s for s in skills if s in text]

# ---------- SCORING ----------
def ats_score(resume, role, req, opt, min_exp):
    exp = extract_experience(resume)

    req_match = skill_match(resume, req)
    opt_match = skill_match(resume, opt)

    req_score = (len(req_match)/len(req))*40 if req else 0
    opt_score = (len(opt_match)/len(opt))*15 if opt else 0

    exp_score = min(exp/min_exp, 1)*25 if min_exp > 0 else 15

    tfidf = TfidfVectorizer(ngram_range=(1,2))
    vectors = tfidf.fit_transform([role, resume])
    sem_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 20

    penalty = 0
    if resume.count("python") > 15:
        penalty += 5
    if len(resume.split()) < 150:
        penalty += 5

    score = min(round(req_score + opt_score + exp_score + sem_score - penalty, 2), 88)

    return score, req_match, opt_match, exp

# ================= UI (ONE SCREEN) =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👔 HR Job Configuration")

    role_text = st.text_area(
        "Role Description",
        height=200
    )

    req_skills = st.text_input(
        "Required Skills (comma separated)",
        "python, sql, nlp"
    )

    opt_skills = st.text_input(
        "Nice-to-Have Skills",
        "aws, docker"
    )

    min_exp = st.number_input(
        "Minimum Experience (years)",
        min_value=0.0,
        step=0.5
    )

with col2:
    st.subheader("📄 Resume Upload")
    resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if resume and role_text:
        if st.button("Evaluate Candidate"):
            with st.spinner("Evaluating resume..."):
                resume_raw = extract_pdf(resume)
                resume_clean = preprocess(resume_raw)
                role_clean = preprocess(role_text)

                score, req_m, opt_m, exp = ats_score(
                    resume_clean,
                    role_clean,
                    [s.strip().lower() for s in req_skills.split(",") if s],
                    [s.strip().lower() for s in opt_skills.split(",") if s],
                    min_exp
                )

            st.success("Evaluation Complete ✅")
            st.metric("Overall Match Score (%)", score)
            st.write("✅ Matched Required Skills:", ", ".join(req_m))
            st.write("❌ Missing Required Skills:",
                     ", ".join(set(req_skills.split(",")) - set(req_m)))
            st.write("⭐ Optional Skills:", ", ".join(opt_m))
            st.write("🕒 Experience Detected:", exp, "years")