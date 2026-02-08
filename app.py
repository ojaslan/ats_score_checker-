import streamlit as st
import fitz
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================= NLTK CLOUD FIX =================
NLTK_PATH = "/tmp/nltk_data"
os.makedirs(NLTK_PATH, exist_ok=True)
nltk.data.path.append(NLTK_PATH)

for pkg in ["punkt", "stopwords", "wordnet"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_PATH)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ================= STREAMLIT CONFIG =================
st.set_page_config("Cloud ATS", layout="wide")
st.title("🧠 Intelligent ATS (LinkedIn-Style Skills)")

# ================= UTIL FUNCTIONS =================
def extract_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join([page.get_text() for page in doc]).lower()

def preprocess(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) > 2
    ]
    return " ".join(tokens)

def extract_experience(text):
    matches = re.findall(r"(\d+\.?\d*)\s+year", text)
    return max([float(x) for x in matches], default=0)

def normalize_skill(skill):
    return skill.lower().strip().replace("-", " ")

def skill_match(text, skills):
    return [s for s in skills if s in text]

# ================= SCORING ENGINE =================
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

    final_score = min(round(req_score + opt_score + exp_score + sem_score - penalty, 2), 88)

    return final_score, req_match, opt_match, exp

# ================= ONE-SCREEN UI =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("👔 HR Job Configuration")

    role_text = st.text_area(
        "Role Description (Responsibilities, tools, domain)",
        height=200
    )

    st.markdown("### 🧩 Required Skills (type & press Enter)")
    required_skills = st.multiselect(
        label="",
        options=[],
        default=[],
        help="Type a skill and press Enter"
    )

    st.markdown("### ⭐ Nice-to-Have Skills")
    optional_skills = st.multiselect(
        label=" ",
        options=[],
        default=[],
        help="Optional skills"
    )

    min_exp = st.number_input(
        "Minimum Experience (years)",
        min_value=0.0,
        step=0.5
    )

with col2:
    st.subheader("📄 Resume Upload")
    resume = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if resume and role_text and required_skills:
        if st.button("🚀 Evaluate Candidate"):
            with st.spinner("Evaluating resume…"):
                resume_raw = extract_pdf(resume)
                resume_clean = preprocess(resume_raw)
                role_clean = preprocess(role_text)

                req_norm = [normalize_skill(s) for s in required_skills]
                opt_norm = [normalize_skill(s) for s in optional_skills]

                score, req_m, opt_m, exp = ats_score(
                    resume_clean,
                    role_clean,
                    req_norm,
                    opt_norm,
                    min_exp
                )

            st.success("Evaluation Complete ✅")

            st.metric("Overall Match Score (%)", score)
            st.write("✅ **Matched Required Skills:**", ", ".join(req_m))
            st.write("❌ **Missing Required Skills:**",
                     ", ".join(set(req_norm) - set(req_m)))
            st.write("⭐ **Matched Optional Skills:**", ", ".join(opt_m))
            st.write("🕒 **Experience Detected:**", exp, "years")