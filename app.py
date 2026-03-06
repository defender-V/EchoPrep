import streamlit as st
import os
import re
import pandas as pd
from dotenv import load_dotenv
from llm_service import InterviewLLMService
from voice_service import VoiceService
from streamlit_mic_recorder import mic_recorder

# Load Env Vars
load_dotenv()

st.set_page_config(
    page_title="EchoPrep | AI Interview Simulator",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Initializing Session State ---
def init_session_state():
    defaults = {
        "messages": [],
        "interview_active": False,
        "questions": [],
        "current_q_index": 0,
        "evaluations": [],
        "parsed_jd": "",
        "theme": "Midnight",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# --- Theme switch (sidebar) ---
with st.sidebar:
    st.markdown("### Appearance")
    st.session_state.theme = st.radio(
        "Theme",
        options=["Midnight", "Pearl"],
        horizontal=True,
        label_visibility="collapsed",
        index=0 if st.session_state.theme == "Midnight" else 1,
    )

themes = {
    "Midnight": {
        "bg_primary": "#0b0b10",
        "bg_secondary": "#14141b",
        "bg_card": "#191923",
        "accent": "#7c3aed",
        "accent_hover": "#9f67ff",
        "accent_muted": "rgba(124, 58, 237, 0.16)",
        "text_primary": "#f8fafc",
        "text_muted": "#a1a1aa",
        "border": "#262638",
        "shadow": "rgba(0,0,0,0.35)",
    },
    "Pearl": {
        "bg_primary": "#f7f7fb",
        "bg_secondary": "#ffffff",
        "bg_card": "#ffffff",
        "accent": "#2563eb",
        "accent_hover": "#3b82f6",
        "accent_muted": "rgba(37, 99, 235, 0.12)",
        "text_primary": "#0f172a",
        "text_muted": "#475569",
        "border": "#e2e8f0",
        "shadow": "rgba(2,6,23,0.10)",
    },
}

t = themes.get(st.session_state.theme, themes["Midnight"])

# --- Custom CSS (theme-aware) ---
css = """
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');
    /* Streamlit uses Material Symbols (ligatures) for icons */
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

    :root {
        --bg-primary: @@bg_primary@@;
        --bg-secondary: @@bg_secondary@@;
        --bg-card: @@bg_card@@;
        --accent: @@accent@@;
        --accent-hover: @@accent_hover@@;
        --accent-muted: @@accent_muted@@;
        --text-primary: @@text_primary@@;
        --text-muted: @@text_muted@@;
        --border: @@border@@;
        --shadow: @@shadow@@;
    }

    /* App background: gradient mesh + subtle noise */
    .stApp {
        background: radial-gradient(1100px 600px at 12% 10%, var(--accent-muted) 0%, transparent 60%),
                    radial-gradient(900px 500px at 88% 18%, rgba(34, 197, 94, 0.10) 0%, transparent 55%),
                    radial-gradient(1200px 650px at 45% 95%, rgba(236, 72, 153, 0.08) 0%, transparent 60%),
                    linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-primary) 100%);
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
            url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='.10'/%3E%3C/svg%3E");
        mix-blend-mode: overlay;
        opacity: 0.35;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'DM Sans', 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }

    /* Default text */
    html, body, [class*="st-"] {
        font-family: 'DM Sans', 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important;
    }

    /* Ensure Streamlit icons render as icons (not text ligatures) */
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
    }

    /* Streamlit-specific icon nodes (prevents "keyboard_double_arrow_left" showing as text) */
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] i,
    [data-testid="stIconMaterial"],
    [data-testid="stIconMaterial"] span {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
        font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        letter-spacing: normal !important;
        text-transform: none !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Chat messages - custom cards */
    [data-testid="stChatMessage"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 10px 28px var(--shadow);
    }
    
    [data-testid="stChatMessage"] p {
        font-family: 'DM Sans', 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
        color: var(--text-primary);
    }
    
    /* Buttons */
    .stButton > button {
        font-family: 'DM Sans', 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif !important;
        font-weight: 500 !important;
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.25rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: var(--accent-hover) !important;
        transform: translateY(-1px);
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.20);
    }
    
    /* Info boxes */
    [data-testid="stAlert"] {
        background: var(--accent-muted) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px;
    }
    
    /* Progress bar container */
    .progress-container {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .progress-label {
        font-family: 'DM Sans', 'Segoe UI', 'Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji', sans-serif;
        color: var(--text-muted);
        font-size: 0.9rem;
    }
    
    /* Score cards in evaluation */
    .score-card {
        background: linear-gradient(135deg, var(--accent-muted) 0%, transparent 100%);
        border: 1px solid var(--accent);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }
    
    .score-badge {
        display: inline-block;
        background: var(--accent);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Landing hero */
    .hero-section {
        text-align: center;
        padding: 3rem 2rem;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        margin: 2rem 0;
        box-shadow: 0 20px 60px var(--shadow);
    }
    
    .hero-section h2 {
        font-size: 1.75rem;
        margin-bottom: 0.5rem;
    }
    
    .hero-section p {
        color: var(--text-muted);
        font-size: 1.1rem;
    }
    
    /* Expander styling (fix icon/text overlap) */
    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        background: var(--bg-card);
        box-shadow: 0 14px 40px var(--shadow);
    }

    [data-testid="stExpander"] summary {
        padding: 0.85rem 1rem !important;
    }

    [data-testid="stExpander"] summary p {
        margin: 0 !important;
    }

    [data-testid="stExpander"] summary > div {
        gap: 0.65rem !important;
        align-items: center !important;
    }

    /* Chart container polish */
    .chart-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.25rem;
        box-shadow: 0 18px 55px var(--shadow);
        margin: 0.25rem 0 1.25rem 0;
    }
</style>
"""

for k, v in {
    "bg_primary": t["bg_primary"],
    "bg_secondary": t["bg_secondary"],
    "bg_card": t["bg_card"],
    "accent": t["accent"],
    "accent_hover": t["accent_hover"],
    "accent_muted": t["accent_muted"],
    "text_primary": t["text_primary"],
    "text_muted": t["text_muted"],
    "border": t["border"],
    "shadow": t["shadow"],
}.items():
    css = css.replace(f"@@{k}@@", str(v))

st.markdown(css, unsafe_allow_html=True)

# Ensure API Key is available before instantiating LLM
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
    st.error("Please add a valid GOOGLE_API_KEY to your `.env` file.")
    st.stop()

# Initialize Backend Service
@st.cache_resource
def get_service():
    return InterviewLLMService()

llm_service = get_service()
voice_service = VoiceService()

# --- Sidebar (Setup Phase) ---
with st.sidebar:
    st.markdown("## 🎤 EchoPrep")
    st.markdown("*AI-Powered Interview Practice*")
    st.divider()
    st.caption("If you see a 429 quota error, reduce questions or try again later.")
    
    st.subheader("Interview Setup")
    jd_input = st.text_area(
        "Paste the Job Description here:",
        height=220,
        placeholder="Paste the full job description to generate tailored interview questions...",
    )
    
    st.markdown("**Interview Options**")
    num_questions = st.slider("Number of questions", min_value=2, max_value=8, value=4)
    interview_type = st.selectbox(
        "Interview focus",
        options=["mixed", "technical", "behavioral"],
        format_func=lambda x: {"mixed": "Mixed (Technical + Behavioral)", "technical": "Technical Only", "behavioral": "Behavioral Only"}[x],
    )
    
    if st.button("🚀 Start Interview", use_container_width=True):
        if not jd_input.strip():
            st.warning("Please provide a Job Description.")
        else:
            with st.spinner("Analyzing JD & generating questions..."):
                parsed_reqs = llm_service.parse_job_description(jd_input)
                st.session_state.parsed_jd = parsed_reqs
                st.session_state.questions = llm_service.generate_interview_plan(
                    parsed_reqs, num_questions=num_questions, interview_type=interview_type
                )
            
            if not st.session_state.questions:
                st.error("Failed to generate questions. Please try again.")
            else:
                # Reset state for a new interview
                st.session_state.interview_active = True
                st.session_state.current_q_index = 0
                st.session_state.evaluations = []
                st.session_state.messages = []
                
                first_q = st.session_state.questions[0]
                first_q_txt = f"Welcome! Let's begin the interview.\n\n**Question 1:** {first_q}"
                audio_fp = voice_service.generate_audio_for_text(first_q_txt)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": first_q_txt,
                    "audio": audio_fp,
                    "autoplay": True,
                })
                st.rerun()
    
    # Show parsed JD when available
    if st.session_state.parsed_jd and st.session_state.interview_active:
        st.divider()
        with st.expander("📋 Extracted Requirements", expanded=False):
            st.markdown(st.session_state.parsed_jd)
    
    # Reset button when interview is done
    if not st.session_state.interview_active and len(st.session_state.evaluations) > 0:
        st.divider()
        if st.button("🔄 Start New Interview", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interview_active = False
            st.session_state.questions = []
            st.session_state.current_q_index = 0
            st.session_state.evaluations = []
            st.rerun()

# --- Main App Interface ---
st.title("🎤 EchoPrep")
st.caption("AI-Driven Interview Simulator — Practice with voice or text")

if not st.session_state.interview_active:
    if len(st.session_state.evaluations) == 0:
        st.markdown("""
        <div class="hero-section">
            <h2>Practice Your Interview Skills</h2>
            <p>Paste a job description in the sidebar to get started. Our AI will analyze it and generate tailored interview questions.</p>
            <p>Answer by speaking or typing — get instant feedback and scores.</p>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 **Get started:** Enter a Job Description in the sidebar and click **Start Interview**.")
else:
    # Progress bar
    total = len(st.session_state.questions)
    current = st.session_state.current_q_index
    progress_pct = (current / total * 100) if total > 0 else 0
    
    st.markdown(f"""
    <div class="progress-container">
        <span class="progress-label">Question {min(current + 1, total)} of {total}</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(progress_pct / 100)
    
    # Display chat history
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg and msg["audio"] is not None:
                should_autoplay = bool(msg.get("autoplay"))
                st.audio(msg["audio"], format="audio/mpeg", autoplay=should_autoplay)
                if should_autoplay:
                    # Prevent replay on every Streamlit rerun
                    st.session_state.messages[i]["autoplay"] = False
    
    # Input section
    st.markdown("---")
    st.markdown("**Your turn** — speak or type your answer below")
    st.caption("⏱️ Keep spoken answers under 60 seconds for best transcription.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.write("🎙️ **Voice**")
        audio_dict = mic_recorder(
            start_prompt="Record",
            stop_prompt="Stop",
            key=f"mic_recorder_{st.session_state.current_q_index}",
            format="wav",
            use_container_width=True,
        )
    with col2:
        st.write("✏️ **Type**")
        text_fallback = st.chat_input("Or type your answer here...")
    
    candidate_answer = None
    
    if text_fallback:
        candidate_answer = text_fallback
    elif audio_dict and "bytes" in audio_dict:
        with st.spinner("Transcribing your audio..."):
            transcribed_text = voice_service.transcribe_audio_buffer(audio_dict["bytes"])
            if transcribed_text and not transcribed_text.startswith("Error"):
                candidate_answer = transcribed_text
            else:
                st.error("Could not transcribe clearly. Please try typing or recording again.")
    
    if candidate_answer:
        # Update state, then rerun so the new question appears ABOVE the input.
        st.session_state.messages.append({"role": "user", "content": candidate_answer})

        with st.spinner("Evaluating your response..."):
            current_q = st.session_state.questions[st.session_state.current_q_index]
            try:
                evaluation = llm_service.evaluate_answer(current_q, candidate_answer)
            except Exception as e:
                st.error(
                    "Evaluation failed (likely API quota / rate limit). "
                    "Try again in a bit or reduce the number of questions."
                )
                evaluation = f"Score: N/A\nFeedback: Evaluation failed. Details: {e}"
            st.session_state.evaluations.append(
                {"question": current_q, "answer": candidate_answer, "evaluation": evaluation}
            )

            st.session_state.current_q_index += 1

            if st.session_state.current_q_index < len(st.session_state.questions):
                next_q = st.session_state.questions[st.session_state.current_q_index]
                q_text = f"**Question {st.session_state.current_q_index + 1}:** {next_q}"
                audio_fp = voice_service.generate_audio_for_text(next_q)
                st.session_state.messages.append(
                    {"role": "assistant", "content": q_text, "audio": audio_fp, "autoplay": True}
                )
            else:
                st.session_state.interview_active = False
                end_msg = "Thank you! The interview is complete. Here's your evaluation report."
                try:
                    audio_fp = voice_service.generate_audio_for_text(end_msg)
                except Exception:
                    audio_fp = None
                st.session_state.messages.append(
                    {"role": "assistant", "content": end_msg, "audio": audio_fp, "autoplay": True}
                )

        st.rerun()

# --- Results View ---
if not st.session_state.interview_active and len(st.session_state.evaluations) > 0:
    st.divider()
    st.subheader("📊 Interview Evaluation Report")
    
    # Overall summary
    scores = []
    score_by_q = []
    for i, eval_data in enumerate(st.session_state.evaluations):
        match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", eval_data["evaluation"], re.IGNORECASE)
        if match:
            s = float(match.group(1))
            scores.append(s)
            score_by_q.append({"Question": f"Q{i+1}", "Score": s})
    
    if scores:
        avg_score = sum(scores) / len(scores)
        st.metric("Overall Score", f"{avg_score:.1f} / 10")

    # Graph: per-question scores
    if score_by_q:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Score by Question")
        df_scores = pd.DataFrame(score_by_q).set_index("Question")
        st.bar_chart(df_scores, height=220)
        st.caption("Tip: Aim for consistent scores across questions, not just one strong answer.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    
    for i, eval_data in enumerate(st.session_state.evaluations):
        q_preview = eval_data["question"][:70] + "..." if len(eval_data["question"]) > 70 else eval_data["question"]
        
        with st.expander(f"Q{i+1}: {q_preview}", expanded=(i == 0)):
            match = re.search(r"Score:\s*(\d+(?:\.\d+)?)", eval_data["evaluation"], re.IGNORECASE)
            score_val = float(match.group(1)) if match else None
            if score_val is not None:
                st.markdown(
                    f"""
                    <div class="score-card">
                        <span class="score-badge">Score: {score_val:g} / 10</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.progress(min(max(score_val / 10.0, 0.0), 1.0))
            st.markdown("**Your Answer**")
            st.info(eval_data["answer"])
            st.markdown("**Feedback & Score**")
            st.markdown(eval_data["evaluation"])
