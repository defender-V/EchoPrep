from langchain_core.prompts import PromptTemplate

# Prompt for parsing the job description
JD_PARSING_PROMPT = PromptTemplate(
    input_variables=["jd_text"],
    template="""
    You are an expert technical recruiter analyzing a job description.
    Extract the core requirements, key technical skills, and essential soft skills from the following job description.

    Job Description:
    {jd_text}

    Return the analysis as a clear, structured list of the top 3-5 most critical skills or requirements for this role.
    Only output the list, no introductory text.
    """
)

# Prompt for generating the interview plan/questions
QUESTION_GENERATION_PROMPT = PromptTemplate(
    input_variables=["parsed_jd", "num_questions", "interview_type"],
    template="""
    Based on the following core requirements and skills extracted from a job description, 
    generate a tailored {num_questions}-question interview plan.
    
    Interview focus: {interview_type}
    - If "technical": Focus on technical skills, problem-solving, and domain knowledge.
    - If "behavioral": Focus on past experiences, teamwork, leadership, and soft skills.
    - If "mixed": Combine both technical and behavioral questions.

    Core Requirements/Skills:
    {parsed_jd}

    Format the output exactly like this, with each question separated by three dashes (---):
    Question 1: [Your first question here]
    ---
    Question 2: [Your second question here]
    ---
    ... (continue for all {num_questions} questions)
    """
)

# Prompt for evaluating an answer
EVALUATION_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""
    You are an expert interviewer evaluating a candidate's response.
    
    Interview Question:
    {question}

    Candidate's Answer:
    {answer}

    Please evaluate the candidate's answer based on clarity, accuracy, and depth.
    Provide your evaluation in the following format exactly:

    Score: [Rating out of 10]
    Feedback: [1-2 sentences explaining the score and how to improve]
    """
)
