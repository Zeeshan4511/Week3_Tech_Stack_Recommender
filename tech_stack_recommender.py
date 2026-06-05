"""
╔══════════════════════════════════════════════════════════════════╗
║         TECH STACK RECOMMENDER — DecodeLabs Project 3            ║
║         AI Recommendation Logic using Content-Based Filtering    ║
╚══════════════════════════════════════════════════════════════════╝

WHAT IS CONTENT-BASED FILTERING?
─────────────────────────────────
Content-Based Filtering recommends items by analyzing the FEATURES
of items and matching them to USER PREFERENCES — no other users needed.
Here: each job role is defined by its required skills (features).
We compare the user's skill-set to each job role's skill-set to find
the best match. This is the same algorithm Netflix uses for "Because
you watched X..."

WHY COSINE SIMILARITY OVER EUCLIDEAN DISTANCE?
───────────────────────────────────────────────
Euclidean distance measures absolute gap in space — it penalizes
length. A user listing 5 skills looks "farther" from a role than
one listing 2 skills, even if the overlap is 100%.

Cosine Similarity measures the ANGLE between vectors, not distance.
Two vectors pointing in the same direction = cos(0°) = 1.0 (perfect
match), regardless of magnitude. This is ideal for skill matching.

Formula: cos(θ) = (A · B) / (||A|| × ||B||)
  A · B   = dot product (shared skill weights)
  ||A||   = magnitude of user vector
  ||B||   = magnitude of job role vector

HOW TF-IDF IMPROVES OVER BINARY VECTORS?
─────────────────────────────────────────
Binary vectors treat all skills equally: Python=1, SQL=1.
TF-IDF (Term Frequency–Inverse Document Frequency) weights skills
by how RARE or DISTINCTIVE they are across all job roles.
A skill like "Python" appears in many roles → low weight (common).
A skill like "Kubernetes" appears in few roles → high weight (rare).
So if you list "Kubernetes", TF-IDF rewards you more for matching
a DevOps role than a generic Data Analyst role. More precise!

COLD START PROBLEM:
────────────────────
When a new user has NO history, how do we recommend? Here, we solve
it at onboarding: the skill inputs ARE the onboarding survey.
If skills don't match anything well, we fall back to:
  (a) Partial matching — at least 1 skill overlaps
  (b) Popularity ranking — most in-demand roles industry-wide
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# ─────────────────────────────────────────────
# DATASET: Job Roles with Associated Skills
# ─────────────────────────────────────────────
JOB_ROLES = {
    "Data Scientist": [
        "Python", "Machine Learning", "SQL", "Statistics", "Data Visualization"
    ],
    "AI Engineer": [
        "Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch"
    ],
    "ML Engineer": [
        "Python", "Machine Learning", "MLOps", "Docker", "Kubernetes"
    ],
    "Data Analyst": [
        "SQL", "Python", "Excel", "Data Visualization", "Statistics"
    ],
    "Backend Developer": [
        "Python", "Java", "SQL", "REST APIs", "Docker"
    ],
    "Frontend Developer": [
        "JavaScript", "React", "CSS", "HTML", "TypeScript"
    ],
    "Full Stack Developer": [
        "JavaScript", "React", "Python", "SQL", "REST APIs"
    ],
    "Cloud Architect": [
        "Cloud Computing", "AWS", "Azure", "Docker", "Kubernetes"
    ],
    "DevOps Engineer": [
        "Docker", "Kubernetes", "CI/CD", "Cloud Computing", "Automation"
    ],
    "Security Analyst": [
        "Cybersecurity", "Networking", "Python", "Linux", "Ethical Hacking"
    ],
    "Database Administrator": [
        "SQL", "PostgreSQL", "Oracle", "Performance Tuning", "Backup & Recovery"
    ],
    "Mobile Developer": [
        "Swift", "Kotlin", "Java", "React Native", "Flutter"
    ],
}

# Popularity scores for cold-start fallback (industry demand index)
POPULARITY = {
    "Data Scientist": 95,
    "AI Engineer": 98,
    "ML Engineer": 93,
    "Cloud Architect": 91,
    "DevOps Engineer": 89,
    "Backend Developer": 88,
    "Full Stack Developer": 87,
    "Data Analyst": 85,
    "Frontend Developer": 84,
    "Security Analyst": 82,
    "Database Administrator": 75,
    "Mobile Developer": 78,
}


# ─────────────────────────────────────────────
# STEP 1: INGESTION — Capture User Skills
# ─────────────────────────────────────────────
def get_user_skills() -> list[str]:
    """
    Prompt the user to enter 3–5 skills.
    Validates input and normalizes casing.
    """
    print("\n" + "─" * 55)
    print("  📥  STEP 1: Tell us your skills")
    print("─" * 55)
    print("  Enter 3 to 5 skills (press Enter after each).")
    print("  Examples: Python, SQL, Machine Learning, Docker,")
    print("            JavaScript, Cloud Computing, React, Java")
    print("─" * 55)

    skills = []
    skill_count = 0
    max_skills = 5

    while skill_count < max_skills:
        required = skill_count < 3
        prompt = f"  Skill {skill_count + 1}{'*' if required else ' (optional)'}: "
        raw = input(prompt).strip()

        if not raw:
            if required:
                print("  ⚠️  This skill is required. Please enter a value.")
                continue
            else:
                # Optional skill skipped
                break

        # Normalize to Title Case
        skill = raw.title()
        if skill in skills:
            print(f"  ⚠️  '{skill}' already added. Try a different skill.")
            continue

        skills.append(skill)
        skill_count += 1

        if skill_count == 3:
            print("\n  ✅  Minimum reached! Press Enter to skip remaining skills.")

    return skills


# ─────────────────────────────────────────────
# STEP 2: VECTORIZATION
# ─────────────────────────────────────────────
def build_corpus(user_skills: list[str]) -> tuple[list[str], list[str]]:
    """
    Build a corpus of skill-documents:
    - One document per job role (skills joined as a string)
    - One document for the user's input
    Returns (corpus, role_names)
    """
    role_names = list(JOB_ROLES.keys())
    role_docs = [" ".join(skills) for skills in JOB_ROLES.values()]
    user_doc = " ".join(user_skills)

    corpus = role_docs + [user_doc]
    return corpus, role_names


def compute_similarity_tfidf(corpus: list[str], role_names: list[str]) -> pd.DataFrame:
    """
    SCORING via TF-IDF + Cosine Similarity.

    TF-IDF Vectorizer converts each skill-document into a weighted
    numeric vector. Skills rare across job roles get higher weights.
    cosine_similarity then calculates the angle between the user
    vector and each job role vector.
    """
    vectorizer = TfidfVectorizer(token_pattern=r"[^,\s]+")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # User vector is the last row; job roles are all rows before it
    job_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]

    scores = cosine_similarity(user_vector, job_vectors).flatten()
    return pd.DataFrame({"Role": role_names, "TF-IDF Score": scores})


def compute_similarity_binary(user_skills: list[str], role_names: list[str]) -> pd.DataFrame:
    """
    SCORING via Binary Vectors + Cosine Similarity.

    Each skill is a dimension. 1 if present, 0 if not.
    Simpler but ignores skill rarity/importance.
    """
    # Build universal vocabulary
    all_skills = set(user_skills)
    for skills in JOB_ROLES.values():
        all_skills.update(skills)
    vocab = sorted(all_skills)

    # User binary vector
    user_vec = np.array([1 if s in user_skills else 0 for s in vocab], dtype=float)

    scores = []
    for role, role_skills in JOB_ROLES.items():
        role_vec = np.array([1 if s in role_skills else 0 for s in vocab], dtype=float)
        # Cosine Similarity: (A·B) / (||A|| × ||B||)
        dot = np.dot(user_vec, role_vec)
        norm = np.linalg.norm(user_vec) * np.linalg.norm(role_vec)
        sim = dot / norm if norm != 0 else 0.0
        scores.append(sim)

    return pd.DataFrame({"Role": role_names, "Binary Score": scores})


# ─────────────────────────────────────────────
# STEP 3 & 4: SORTING + TOP-N FILTERING
# ─────────────────────────────────────────────
def rank_and_filter(df_tfidf: pd.DataFrame, df_binary: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Merge both scoring DataFrames, sort by TF-IDF score (primary),
    and return top N results.
    """
    merged = df_tfidf.merge(df_binary, on="Role")
    merged = merged.sort_values("TF-IDF Score", ascending=False).reset_index(drop=True)
    return merged.head(top_n)


# ─────────────────────────────────────────────
# COLD START HANDLER
# ─────────────────────────────────────────────
def handle_cold_start(user_skills: list[str], top_n: int = 3) -> None:
    """
    Cold Start: No strong similarity found.
    Strategy A — Partial matching (at least 1 skill overlaps).
    Strategy B — Fall back to popularity ranking.
    """
    print("\n  🌡️  Low similarity detected. Applying Cold-Start strategies...")
    print("  ─────────────────────────────────────────────────────────")

    # Strategy A: Partial matching
    partial_matches = []
    for role, role_skills in JOB_ROLES.items():
        matched = [s for s in user_skills if s in role_skills]
        if matched:
            partial_matches.append((role, matched))

    if partial_matches:
        print("\n  📌  STRATEGY A — Partial Skill Matches:\n")
        for i, (role, matched) in enumerate(partial_matches[:top_n], 1):
            print(f"  {i}. {role}")
            print(f"     ✓ Partial match on: {', '.join(matched)}")
        return

    # Strategy B: Popularity fallback
    print("\n  📊  STRATEGY B — Trending Roles (No skill match found):\n")
    popular = sorted(POPULARITY.items(), key=lambda x: x[1], reverse=True)
    for i, (role, score) in enumerate(popular[:top_n], 1):
        print(f"  {i}. {role}  (Industry Demand Index: {score}/100)")

    print("\n  💡  Tip: Try skills like Python, SQL, JavaScript, Docker,")
    print("      or Cloud Computing for better matches.")


# ─────────────────────────────────────────────
# OUTPUT FORMATTER
# ─────────────────────────────────────────────
def display_results(top_roles: pd.DataFrame, user_skills: list[str]) -> None:
    """
    Render the top-3 recommendations with match %, matching skills,
    and a binary vs TF-IDF comparison table.
    """
    COLD_START_THRESHOLD = 0.05  # below 5% → cold start

    if top_roles["TF-IDF Score"].max() < COLD_START_THRESHOLD:
        handle_cold_start(user_skills)
        return

    print("\n" + "═" * 55)
    print("  🔍  RECOMMENDED JOB ROLES  (TF-IDF + Cosine Similarity)")
    print("═" * 55)

    medals = ["🥇", "🥈", "🥉"]

    for i, row in top_roles.iterrows():
        role = row["Role"]
        tfidf_pct = round(row["TF-IDF Score"] * 100, 1)
        binary_pct = round(row["Binary Score"] * 100, 1)
        role_skills = JOB_ROLES[role]
        matching = [s for s in user_skills if s in role_skills]
        missing = [s for s in role_skills if s not in user_skills]

        print(f"\n  {medals[i]}  {role}")
        print(f"      TF-IDF Match : {'█' * int(tfidf_pct // 5):<20} {tfidf_pct}%")
        print(f"      Binary Match : {'░' * int(binary_pct // 5):<20} {binary_pct}%")
        if matching:
            print(f"      ✓ Matching   : {', '.join(matching)}")
        if missing:
            print(f"      ➕ To learn  : {', '.join(missing[:2])}")

    # Comparison table
    print("\n" + "─" * 55)
    print("  📊  Binary vs TF-IDF Score Comparison")
    print("─" * 55)
    print(f"  {'Role':<26} {'Binary':>8}  {'TF-IDF':>8}")
    print("  " + "─" * 50)
    for _, row in top_roles.iterrows():
        print(f"  {row['Role']:<26} {row['Binary Score']*100:>7.1f}%  {row['TF-IDF Score']*100:>7.1f}%")
    print("─" * 55)
    print("  💡  TF-IDF weights rare skills higher → more accurate")
    print("─" * 55)


# ─────────────────────────────────────────────
# MAIN: IPO PIPELINE
# ─────────────────────────────────────────────
def run_recommender() -> None:
    print("\n" + "╔" + "═" * 53 + "╗")
    print("║      🤖  TECH STACK RECOMMENDER — DecodeLabs      ║")
    print("║         AI-Powered Career Path Matching            ║")
    print("╚" + "═" * 53 + "╝")
    print("  Powered by: TF-IDF Vectorization + Cosine Similarity")
    print("  Pipeline:   Ingest → Score → Sort → Filter (Top-3)")

    while True:
        # ── INPUT ──────────────────────────────────
        user_skills = get_user_skills()

        if not user_skills:
            print("  ⚠️  No skills entered. Please try again.")
            continue

        print(f"\n  ✅  Skills captured: {', '.join(user_skills)}")

        # ── PROCESS ────────────────────────────────
        print("\n  ⚙️  Processing...")
        corpus, role_names = build_corpus(user_skills)
        df_tfidf = compute_similarity_tfidf(corpus, role_names)
        df_binary = compute_similarity_binary(user_skills, role_names)
        top_roles = rank_and_filter(df_tfidf, df_binary, top_n=3)

        # ── OUTPUT ─────────────────────────────────
        display_results(top_roles, user_skills)

        # ── LOOP CONTROL ───────────────────────────
        print("\n  Type 'quit' to exit or press Enter to try again.")
        again = input("  > ").strip().lower()
        if again == "quit":
            print("\n  👋  Thanks for using the Tech Stack Recommender!")
            print("  📁  Commit your work: git add . && git commit -m 'Project 3: Tech Stack Recommender'")
            print()
            break


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        run_recommender()
    except KeyboardInterrupt:
        print("\n\n  ⛔  Session interrupted. Goodbye!\n")
        sys.exit(0)
