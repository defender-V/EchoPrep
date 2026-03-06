import streamlit as st
import os
import time
from dotenv import load_dotenv
from llm_service import InterviewLLMService
from voice_service import VoiceService
from streamlit_mic_recorder import mic_recorder

# Load Env Vars
load_dotenv()

st.set_page_config(page_title="AI Interview Simulator", page_icon="🎤")

# Ensure API Key is available before instantiating LLM
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "your_api_key_here":
    st.error("Please add a valid GOOGLE_API_KEY to your `.env` file.")
    st.stop()

# Initialize Backend Service
def get_service():
    return InterviewLLMService()

llm_service = get_service()
voice_service = VoiceService()

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
            first_q_txt = f"Welcome! Let's begin the interview.\n\n**Question 1:** {first_q}"
            
            # Generate the audio for the very first question upon starting
            audio_fp = voice_service.generate_audio_for_text(first_q_txt)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": first_q_txt,
                "audio": audio_fp
            })


# --- Main App Interface ---
st.title("AI-Driven Interview Simulator")

if not st.session_state.interview_active:
    st.info("👈 Please enter a Job Description in the sidebar to start the interview.")
else:
    # 1. Display chat history FIRST (so it sits above the input)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "audio" in msg and msg["audio"] is not None:
                # Replay audio if it exists in history
                st.audio(msg["audio"], format="audio/mpeg", autoplay=False)
    
    # 2. Accept User Input Let's take the candidate's answer
    
    # We provide a fallback text input just in case, but emphasize the mic
    st.info("⏱️ **Time Limit:** Please keep your spoken answers under 60 seconds.")
    col1, col2 = st.columns([1, 4])
    with col1:
        st.write("🎙️ **Speak Answer:**")
        # The recorder returns a dictionary with 'bytes' when recording stops
        audio_dict = mic_recorder(
            start_prompt="Record",
            stop_prompt="Stop",
            key=f"mic_recorder_{st.session_state.current_q_index}", # Give it a unique key per question so it resets
            format="wav",
            use_container_width=True
        )
    with col2:
        text_fallback = st.chat_input("Or type your answer here...")

    candidate_answer = None

    # Determine if we got text from the fallback or audio from the mic
    if text_fallback:
        candidate_answer = text_fallback
    elif audio_dict and "bytes" in audio_dict:
        with st.spinner("Transcribing your audio..."):
            transcribed_text = voice_service.transcribe_audio_buffer(audio_dict["bytes"])
            if transcribed_text and not transcribed_text.startswith("Error"):
                candidate_answer = transcribed_text
            else:
                st.error("Could not hear or transcribe you clearly. Please try typing or recording again.")
    
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
                
                # Generate audio for the next question
                audio_fp = voice_service.generate_audio_for_text(next_q)
                
                # We store both text and the TTS buffer in state
                st.session_state.messages.append({"role": "assistant", "content": q_text, "audio": audio_fp})
                
                with st.chat_message("assistant"):
                    st.markdown(q_text)
                    st.audio(audio_fp, format="audio/mpeg", autoplay=True)
            else:
                # Interview is over
                st.session_state.interview_active = False
                end_msg = "Thank you! The interview is now complete. Generating your evaluation report..."
                audio_fp = voice_service.generate_audio_for_text(end_msg)
                
                st.session_state.messages.append({"role": "assistant", "content": end_msg, "audio": audio_fp})
                
                with st.chat_message("assistant"):
                    st.markdown(end_msg)
                    st.audio(audio_fp, format="audio/mpeg", autoplay=True)
                
                # Sleep briefly so the user can hear the TTS finish playing before the screen refreshes to the results
                time.sleep(4)
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
