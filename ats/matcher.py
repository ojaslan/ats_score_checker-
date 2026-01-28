import re

def clean_text(text):
    return re.sub(r"[^a-zA-Z ]", " ", text.lower())

def match_skills(resume_text, skills):
    resume_words = clean_text(resume_text).split()
    matched = [skill for skill in skills if skill.lower() in resume_words]
    missing = list(set(skills) - set(matched))
    return matched, missing
