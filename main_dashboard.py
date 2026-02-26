# main.py - Resume Genie (Optimized Fast Version)

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────


st.set_page_config(
    page_title="Resume Genie",
    page_icon="🚀",
    layout="wide"
)

OPENAI_API_KEY = "API_KEY"

# ───────────────────────────────────────────────
# LIGHTWEIGHT CSS (FAST)
# ───────────────────────────────────────────────
st.markdown("""
<style>

.block-container {
    padding-top: 1.5rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 100%;
}

/* Smooth gradient but NO heavy animation */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Hero */
.hero {
    text-align: center;
    padding: 2rem 1rem;
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg,#818cf8,#a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Cards */
.card {
    padding: 2rem;
    border-radius: 16px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 2rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: white;
    border-radius: 10px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    opacity: 0.9;
}

</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# HERO
# ───────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Resume Genie 🚀</h1>
    <p>AI-powered tools to optimize your resume and accelerate your career</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("⚡ Faster Applications", "5x")
col2.metric("🎯 ATS Optimization", "95%+")
col3.metric("🤖 AI Powered", "GPT-4o-mini")

# ───────────────────────────────────────────────
# MODEL (CACHED)
# ───────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.2,
        max_tokens=1000
    )

llm = get_llm()

# ───────────────────────────────────────────────
# PDF LOADER
# ───────────────────────────────────────────────
@st.cache_data
def extract_resume_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs)
        return text[:6000]
    finally:
        os.unlink(tmp_path)

# ───────────────────────────────────────────────
# PROMPTS
# ───────────────────────────────────────────────
COVER_PROMPT = PromptTemplate.from_template("""
Write a professional cover letter (300–400 words).
Match resume with job description.
Do not invent facts.

Job:
{job_description}

Resume:
{resume_text}
""")

SCORER_PROMPT = PromptTemplate.from_template("""
Provide format:

Score: X/100
Match: X%
Keywords matched: ...
Missing keywords: ...
Improvements: ...

Job:
{job_description}

Resume:
{context}
""")

CHECKER_PROMPT = PromptTemplate.from_template("""
Structure:

Score: X/100
Strengths: ...
Weaknesses: ...
Recommended Skills: ...

Resume:
{context}
""")

# ───────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────
st.sidebar.title("🧠 Resume Genie")
tool = st.sidebar.radio("Choose Tool", [
    "Cover Letter",
    "Resume Matcher",
    "Resume Checker",
    "Career Coach"
])

# ───────────────────────────────────────────────
# TOOLS
# ───────────────────────────────────────────────
if tool == "Cover Letter":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    jd = st.text_area("Paste Job Description")
    file = st.file_uploader("Upload Resume PDF", type="pdf")

    if file and jd and st.button("Generate"):
        resume = extract_resume_text(file)
        chain = COVER_PROMPT | llm
        result = chain.invoke({
            "job_description": jd,
            "resume_text": resume
        })
        st.markdown(result.content)

    st.markdown('</div>', unsafe_allow_html=True)

elif tool == "Resume Matcher":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    jd = st.text_area("Paste Job Description")
    file = st.file_uploader("Upload Resume PDF", type="pdf")

    if file and jd and st.button("Analyze"):
        resume = extract_resume_text(file)
        chain = SCORER_PROMPT | llm
        result = chain.invoke({
            "job_description": jd,
            "context": resume
        })
        st.markdown(result.content)

    st.markdown('</div>', unsafe_allow_html=True)

elif tool == "Resume Checker":

    st.markdown('<div class="card">', unsafe_allow_html=True)

    file = st.file_uploader("Upload Resume PDF", type="pdf")

    if file and st.button("Evaluate"):
        resume = extract_resume_text(file)
        chain = CHECKER_PROMPT | llm
        result = chain.invoke({"context": resume})
        st.markdown(result.content)

    st.markdown('</div>', unsafe_allow_html=True)

elif tool == "Career Coach":

    if "history" not in st.session_state:
        st.session_state.history = []

    file = st.file_uploader("Upload Resume First", type="pdf")

    if file:
        resume = extract_resume_text(file)

        # Show previous chat messages
        for msg in st.session_state.history:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(msg.content)

        # Chat input
        question = st.chat_input("Ask career question...")

        if question:

            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(question)

            st.session_state.history.append(
                HumanMessage(content=question)
            )

            # System instruction
            system = SystemMessage(content=f"""
You are an experienced AI career coach.
Be practical, concise, and actionable.
Avoid generic motivation.

Resume:
{resume}
""")

            response = llm.invoke([system] + st.session_state.history)

            # Show assistant reply
            with st.chat_message("assistant"):
                st.markdown(response.content)

            st.session_state.history.append(
                AIMessage(content=response.content)
            )

st.markdown("---")
st.caption("Resume Genie © 2026")