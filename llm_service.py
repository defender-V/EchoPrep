import os
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import JD_PARSING_PROMPT, QUESTION_GENERATION_PROMPT, EVALUATION_PROMPT

class InterviewLLMService:
    def __init__(self):
        # We rely on the python-dotenv loading the api key in app.py
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_api_key_here":
            raise ValueError("Google API key is missing or invalid. Please update your .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", 
            temperature=0.7,
            google_api_key=api_key
        )

    def parse_job_description(self, jd_text: str) -> str:
        """Extracts core requirements from a JD."""
        chain =  JD_PARSING_PROMPT | self.llm
        response = chain.invoke({"jd_text": jd_text})
        return response.content.strip()

    def generate_interview_plan(self, parsed_jd: str) -> list[str]:
        """Generates a list of interview questions based on parsed requirements."""
        chain = QUESTION_GENERATION_PROMPT | self.llm
        response = chain.invoke({"parsed_jd": parsed_jd})
        
        # Parse the output separated by "---"
        raw_text = response.content.strip()
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

    def evaluate_answer(self, question: str, answer: str) -> str:
        """Evaluates a candidate's answer and returns a score/feedback."""
        chain = EVALUATION_PROMPT | self.llm
        response = chain.invoke({"question": question, "answer": answer})
        return response.content.strip()
