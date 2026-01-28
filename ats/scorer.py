def ats_score(matched, total_skills):
    if total_skills == 0:
        return 0
    return round((len(matched) / total_skills) * 100, 2)
