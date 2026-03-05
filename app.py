import streamlit as st
import os
from dotenv import load_dotenv
from llm_service import InterviewLLMService

# Load Env Vars
load_dotenv()

st.set_page_config(page_title="AI Interview Simulator", page_icon="🎤")

# Ensure API Key is available before instantiating LLM
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
    st.error("Please add a valid GOOGLE_API_KEY to your `.env` file.")
    st.stop()

# Initialize Backend Service
@st.cache_resource
def get_service():
    return InterviewLLMService()

llm_service = get_service()

# --- Initializing Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # Chat history
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "questions" not in st.session_state:
    st.session_state.questions = []
if "current_q_index" not in st.session_state:
    st.session_state.current_q_index = 0
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

# --- Sidebar (Setup Phase) ---
with st.sidebar:
    st.title("Interview Setup")
    jd_input = st.text_area("Paste the Job Description here:", height=300)
    
    if st.button("Start Interview"):
        if not jd_input.strip():
            st.warning("Please provide a Job Description.")
        else:
            with st.spinner("Analyzing JD & generating questions..."):
                parsed_reqs = llm_service.parse_job_description(jd_input)
                st.session_state.questions = llm_service.generate_interview_plan(parsed_reqs)
                
            # Reset state for a new interview
            st.session_state.interview_active = True
            st.session_state.current_q_index = 0
            st.session_state.evaluations = []
            
            # Formulate first prompt
            st.session_state.messages = []
            first_q = st.session_state.questions[0]
            st.session_state.messages.append({"role": "assistant", "content": f"Welcome! Let's begin the interview.\n\n**Question 1:** {first_q}"})


# --- Main App Interface ---
st.title("AI-Driven Interview Simulator")

if not st.session_state.interview_active:
    st.info("👈 Please enter a Job Description in the sidebar to start the interview.")
else:
    # 1. Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 2. Accept User Input Let's take the candidate's answer
    candidate_answer = st.chat_input("Type your answer here...")
    
    if candidate_answer:
        # Show candidate's answer
        st.session_state.messages.append({"role": "user", "content": candidate_answer})
        with st.chat_message("user"):
            st.markdown(candidate_answer)
            
        with st.spinner("Evaluating your response..."):
            # Get current question
            current_q = st.session_state.questions[st.session_state.current_q_index]
            
            # Evaluate it
            evaluation = llm_service.evaluate_answer(current_q, candidate_answer)
            st.session_state.evaluations.append({
                "question": current_q,
                "answer": candidate_answer,
                "evaluation": evaluation
            })
            
            # Advance to the next question
            st.session_state.current_q_index += 1
            
            if st.session_state.current_q_index < len(st.session_state.questions):
                # Ask the next question
                next_q = st.session_state.questions[st.session_state.current_q_index]
                q_text = f"**Question {st.session_state.current_q_index + 1}:** {next_q}"
                st.session_state.messages.append({"role": "assistant", "content": q_text})
                with st.chat_message("assistant"):
                    st.markdown(q_text)
            else:
                # Interview is over
                st.session_state.interview_active = False
                st.session_state.messages.append({"role": "assistant", "content": "Thank you! The interview is now complete. Generating your evaluation report..."})
                with st.chat_message("assistant"):
                    st.markdown("Thank you! The interview is now complete. Generating your evaluation report...")
                st.rerun() # Refresh to show results

# --- Results View ---
if not st.session_state.interview_active and len(st.session_state.evaluations) > 0:
    st.divider()
    st.subheader("📊 Interview Evaluation Report")
    
    for i, eval_data in enumerate(st.session_state.evaluations):
        with st.expander(f"Q{i+1}: {eval_data['question'][:80]}...", expanded=True):
            st.markdown(f"**Your Answer:**\n{eval_data['answer']}")
            st.markdown("---")
            st.markdown(f"**Feedback & Score:**\n{eval_data['evaluation']}")
