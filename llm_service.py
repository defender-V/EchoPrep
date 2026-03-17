import os
import re
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import JD_PARSING_PROMPT, QUESTION_GENERATION_PROMPT, EVALUATION_PROMPT

class InterviewLLMService:
    def __init__(self):
        # We rely on the python-dotenv loading the api key in app.py
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("Google API key is missing or invalid. Please update your .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.7,
            google_api_key=api_key
        )

    def _content_to_str(self, content) -> str:
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    text_parts.append(item["text"])
                else:
                    text_parts.append(str(item))
            content = " ".join(text_parts)
        return str(content).strip()

    def _retry_delay_seconds(self, msg: str) -> float | None:
        # Common formats observed in error messages
        m = re.search(r"retry in\s+([0-9.]+)s", msg, re.IGNORECASE)
        if m:
            return float(m.group(1))
        m = re.search(r"retryDelay'\s*:\s*'(\d+)s'", msg)
        if m:
            return float(m.group(1))
        return None

    def _invoke_with_retry(self, chain, payload: dict, max_attempts: int = 3):
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                return chain.invoke(payload)
            except Exception as e:
                last_err = e
                msg = str(e)
                is_quota = ("RESOURCE_EXHAUSTED" in msg) or ("429" in msg) or ("quota" in msg.lower())
                if not is_quota or attempt == max_attempts:
                    raise
                delay = self._retry_delay_seconds(msg)
                time.sleep((delay if delay is not None else 2.0) + 0.25 * attempt)
        raise last_err  # pragma: no cover

    def parse_job_description(self, jd_text: str) -> str:
        """Extracts core requirements from a JD."""
        chain = JD_PARSING_PROMPT | self.llm
        try:
            response = self._invoke_with_retry(chain, {"jd_text": jd_text})
            return self._content_to_str(response.content)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                return "Rate limit/quota exceeded while analyzing the JD. Please wait a bit and try again, or use a different API key / billing plan."
            raise

    def generate_interview_plan(self, parsed_jd: str, num_questions: int = 3, interview_type: str = "mixed") -> list[str]:
        """Generates a list of interview questions based on parsed requirements."""
        chain = QUESTION_GENERATION_PROMPT | self.llm
        try:
            response = self._invoke_with_retry(chain, {
                "parsed_jd": parsed_jd,
                "num_questions": num_questions,
                "interview_type": interview_type,
            })
            content = self._content_to_str(response.content)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                # Graceful fallback so the app can still run
                return [
                    "Walk me through your most relevant experience for this role.",
                    "Describe a challenging problem you solved and how you approached it.",
                    "How do you prioritize tasks and communicate progress with your team?",
                ][: max(2, min(num_questions, 8))]
            raise
            
        # Parse the output separated by "---"
        raw_text = content.strip()
        questions = []
        for block in raw_text.split("---"):
            block = block.strip()
            if block:
                # Remove "Question X: " prefix if present to clean it up
                if ":" in block and block.lower().startswith("question"):
                    block = block.split(":", 1)[1].strip()
                questions.append(block)
                
        # Fallback if parsing fails or LLM hallucinates format
        if not questions:
            questions = ["Describe your relevant experience.", "How do you handle technical challenges?"]
            
        return questions

    def evaluate_answer(self, question: str, answer: str, vibe_metrics: str = "N/A (Text input only)") -> str:
        """Evaluates a candidate's answer and returns a score/feedback."""
        chain = EVALUATION_PROMPT | self.llm
        try:
            response = self._invoke_with_retry(chain, {
                "question": question, 
                "answer": answer,
                "vibe_metrics": vibe_metrics
            })
            return self._content_to_str(response.content)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                return (
                    "Score: N/A\n"
                    "Feedback: Gemini API rate limit/quota exceeded. Wait a bit and try again, or upgrade/enable billing for your API key."
                )
            raise
