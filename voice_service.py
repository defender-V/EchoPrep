import streamlit as st
import os
import io
import time
from gtts import gTTS
import speech_recognition as sr
from io import BytesIO

class VoiceService:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def generate_audio_for_text(self, text: str) -> BytesIO:
        """
        Takes raw text (e.g., from the AI interviewer) and converts it to a playable
        audio buffer using Google Text-to-Speech (gTTS).
        """
        # Strip markdown asterisks and hashtags so the automated voice doesn't read them aloud
        clean_text = text.replace('**', '').replace('*', '').replace('#', '')
        
        tts = gTTS(text=clean_text, lang='en', slow=False)
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp

    def transcribe_audio_buffer(self, audio_bytes: bytes) -> str:
        """
        Takes recorded audio bytes (from the Streamlit mic recorder),
        prepares it via an AudioFile, and uses SpeechRecognition to transcribe.
        Note: The mic recorder usually outputs WAV bytes directly.
        """
        try:
            # We wrap the raw bytes in a BytesIO object so SpeechRecognition can read it as a file
            audio_data_io = BytesIO(audio_bytes)
            
            with sr.AudioFile(audio_data_io) as source:
                audio = self.recognizer.record(source)
                
            # Using the free Google Web Speech API for simplicity
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return ""  # Audio was unintelligible
        except sr.RequestError as e:
            st.error(f"Could not request results from Speech Recognition service; {e}")
            return "Error: STT Service Unavailable."
        except Exception as e:
            st.error(f"Error processing audio: {e}")
            return "Error: Could not process audio."
