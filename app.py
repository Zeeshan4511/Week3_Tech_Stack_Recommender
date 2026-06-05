import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
# DATA
# ======================================================

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
    "Mobile Developer": 78,
    "Database Administrator": 75,
}

# ======================================================
# FUNCTIONS
# ======================================================

def build_corpus(user_skills):
    role_names = list(JOB_ROLES.keys())
    role_docs = [" ".join(skills) for skills in JOB_ROLES.values()]
    user_doc = " ".join(user_skills)

    return role_docs + [user_doc], role_names


def compute_similarity_tfidf(corpus, role_names):
    vectorizer = TfidfVectorizer(token_pattern=r"[^,\s]+")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    job_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]

    scores = cosine_similarity(user_vector, job_vectors).flatten()

    return pd.DataFrame({
        "Role": role_names,
        "TF-IDF Score": scores
    })


def compute_similarity_binary(user_skills, role_names):
    all_skills = set(user_skills)

    for skills in JOB_ROLES.values():
        all_skills.update(skills)

    vocab = sorted(all_skills)

    user_vec = np.array(
        [1 if skill in user_skills else 0 for skill in vocab],
        dtype=float
    )

    scores = []

    for role, role_skills in JOB_ROLES.items():
        role_vec = np.array(
            [1 if skill in role_skills else 0 for skill in vocab],
            dtype=float
        )

        dot = np.dot(user_vec, role_vec)
        norm = np.linalg.norm(user_vec) * np.linalg.norm(role_vec)

        sim = dot / norm if norm != 0 else 0
        scores.append(sim)

    return pd.DataFrame({
        "Role": role_names,
        "Binary Score": scores
    })


def rank_and_filter(df_tfidf, df_binary, top_n=3):
    merged = df_tfidf.merge(df_binary, on="Role")
    merged = merged.sort_values(
        "TF-IDF Score",
        ascending=False
    ).reset_index(drop=True)

    return merged.head(top_n)


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Tech Stack Recommender",
    page_icon="🤖",
    layout="wide"
)

# ======================================================
# HEADER
# ======================================================

st.title("🤖 Tech Stack Recommender")
st.markdown(
    """
    Discover the best tech career paths using
    **TF-IDF Vectorization + Cosine Similarity**
    """
)

st.divider()

# ======================================================
# INPUT
# ======================================================

st.subheader("🛠 Enter Your Skills")

skills_input = st.text_input(
    "Enter 3–5 skills separated by commas",
    placeholder="Python, SQL, Machine Learning, Docker"
)

if st.button("🚀 Recommend Roles", use_container_width=True):

    user_skills = [
        skill.strip().title()
        for skill in skills_input.split(",")
        if skill.strip()
    ]

    if len(user_skills) < 3:
        st.error("Please enter at least 3 skills.")
        st.stop()

    corpus, role_names = build_corpus(user_skills)

    df_tfidf = compute_similarity_tfidf(
        corpus,
        role_names
    )

    df_binary = compute_similarity_binary(
        user_skills,
        role_names
    )

    top_roles = rank_and_filter(
        df_tfidf,
        df_binary
    )

    threshold = 0.05

    # ==================================================
    # COLD START
    # ==================================================

    if top_roles["TF-IDF Score"].max() < threshold:

        st.warning("Low similarity detected.")

        st.subheader("🔥 Trending Career Roles")

        popular = sorted(
            POPULARITY.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for role, score in popular[:5]:
            st.write(f"**{role}** — Demand Index: {score}/100")

        st.stop()

    # ==================================================
    # RESULTS
    # ==================================================

    st.subheader("🏆 Recommended Roles")

    medals = ["🥇", "🥈", "🥉"]

    for i, row in top_roles.iterrows():

        role = row["Role"]
        tfidf_score = row["TF-IDF Score"]
        binary_score = row["Binary Score"]

        role_skills = JOB_ROLES[role]

        matching = [
            skill
            for skill in user_skills
            if skill in role_skills
        ]

        missing = [
            skill
            for skill in role_skills
            if skill not in user_skills
        ]

        with st.container(border=True):

            st.markdown(
                f"### {medals[i]} {role}"
            )

            st.write(
                f"**TF-IDF Match:** {tfidf_score*100:.1f}%"
            )
            st.progress(float(tfidf_score))

            st.write(
                f"**Binary Match:** {binary_score*100:.1f}%"
            )

            if matching:
                st.success(
                    f"Matching Skills: {', '.join(matching)}"
                )

            if missing:
                st.info(
                    f"Skills to Learn: {', '.join(missing[:3])}"
                )

    # ==================================================
    # COMPARISON TABLE
    # ==================================================

    st.subheader("📊 Score Comparison")

    table_df = top_roles.copy()

    table_df["TF-IDF Score"] = (
        table_df["TF-IDF Score"] * 100
    ).round(2)

    table_df["Binary Score"] = (
        table_df["Binary Score"] * 100
    ).round(2)

    st.dataframe(
        table_df,
        use_container_width=True
    )

    st.caption(
        "TF-IDF gives higher importance to rare skills, producing more accurate recommendations."
    )