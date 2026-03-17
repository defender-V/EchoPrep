import streamlit as st
from streamlit_mic_recorder import mic_recorder
from vibe_service import VibeService
from voice_service import VoiceService

st.set_page_config(page_title="Vibe Check Tester", page_icon="🎤")

st.title("🎤 Test Your Voice Vibe")
st.markdown("Record a snippet of your voice snippet here. We'll transcribe it and run it directly through the PyTorch Audio ML Model!")

@st.cache_resource
def get_vibe_service():
    return VibeService()

@st.cache_resource
def get_voice_service():
    return VoiceService()

vibe_service = get_vibe_service()
voice_service = get_voice_service()

st.divider()
st.subheader("1. Record your Voice")
st.write("Say something like: 'Hello, I am testing the speech emotion recognition model today!'")

audio_dict = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    format="wav",
    use_container_width=True,
    key="vibe_tester"
)

if audio_dict and "bytes" in audio_dict:
    st.divider()
    st.subheader("2. Results")
    with st.spinner("Analyzing audio..."):
        # Transcribe
        text = voice_service.transcribe_audio_buffer(audio_dict["bytes"])
        st.write("**Transcription:**", text)
        
        # ML Analysis
        if text and not text.startswith("Error"):
            metrics = vibe_service.analyze_vibe(text, audio_dict["bytes"])
            st.info(metrics)
        else:
            st.error("Audio was unclear. Please try recording again.")
