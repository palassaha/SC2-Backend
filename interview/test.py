import os
import json
from groq import Groq
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv

load_dotenv()

def get_top_interview_questions(payload: dict) -> dict:
    """
    Uses Groq's llama-3.1-8b-instant model + DuckDuckGo search
    to fetch top 4-5 interview questions asked at a company
    for a specific job role.
    
    Args:
        payload (dict): { "company name": str, "job role": str }
    
    Returns:
        dict: {
            "company": str,
            "job_role": str,
            "top_questions": List[str]
        }
    """
    company = payload.get("company name")
    role = payload.get("job role")
    if not company or not role:
        raise ValueError("Payload must contain 'company name' and 'job role'")
    
    # Initialize Groq client
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    search = DuckDuckGoSearchRun()
    
    # Step 1: Perform web search
    query = f"{company} {role} interview questions site:glassdoor.com OR site:ambitionbox.com"
    search_results = search.run(query)
    
    # Step 2: Ask Groq model to extract questions
    prompt = f"""
    You are given some raw web search results about interview questions for
    {company} - {role}. Extract only the top 4-5 distinct interview questions
    that are most relevant. Return them as a JSON list under the key "top_questions".

    Search Results:
    {search_results}

    Respond in JSON format:
    {{
      "company": "{company}",
      "job_role": "{role}",
      "top_questions": [...]
    }}
    """
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # Step 3: Parse response
    content = response.choices[0].message.content.strip()
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # If model adds extra text, attempt to isolate JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        result = json.loads(content[start:end])
    
    return result


# Example usage
if __name__ == "__main__":
    payload = {"company name": "Onebanc", "job role": "Data Scientist"}
    questions = get_top_interview_questions(payload)
    print(json.dumps(questions, indent=2))
