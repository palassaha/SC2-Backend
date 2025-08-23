from groq import Groq
import base64
import os
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
import PyPDF2
from pathlib import Path

load_dotenv()

def encode_image(image_path: str) -> str:
    """Convert an image file into base64 string for API input."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_image(image_path: str) -> str:
    """Extract text from resume image using Groq Vision model."""
    base64_image = encode_image(image_path)
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    prompt = """
    You are a text extractor. Extract ALL text content from this resume image.
    Return only the extracted text, maintaining the structure and format as much as possible.
    Do not add any commentary or explanations.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
        )
        
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""

def extract_resume_content(file_path: str) -> str:
    """Extract content from resume file (PDF or image)."""
    file_extension = Path(file_path).suffix.lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def match_skills_with_ai(resume_content: str, company_skills: List[str]) -> Dict[str, bool]:
    """
    Use AI to match skills from resume with company required skills.
    
    Args:
        resume_content (str): Extracted text content from resume
        company_skills (List[str]): List of skills required by company
        
    Returns:
        Dict[str, bool]: Dictionary mapping each company skill to True/False
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    skills_str = ", ".join(company_skills)
    
    prompt = f"""
    You are a skills matcher for recruitment.
    
    RESUME CONTENT:
    {resume_content}
    
    COMPANY REQUIRED SKILLS:
    {skills_str}
    
    For each required skill, determine if the candidate has that skill based on their resume.
    Consider:
    - Direct mentions of the skill
    - Related technologies or frameworks
    - Experience that implies the skill
    - Projects that would require the skill
    - Similar or equivalent skills
    
    Return ONLY a JSON object mapping each skill to true/false:
    {{
        "skill1": true,
        "skill2": false,
        "skill3": true
    }}
    
    Be thorough but fair in your assessment.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        content = chat_completion.choices[0].message.content.strip()
        print("Raw AI Response:", content)
        
        # Try to extract JSON from the response
        try:
            # Find JSON in the response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = content[start_idx:end_idx]
                skills_match = json.loads(json_str)
                
                # Ensure all company skills are in the response
                result = {}
                for skill in company_skills:
                    result[skill] = skills_match.get(skill, False)
                
                return result
            else:
                raise ValueError("No JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing AI response: {e}")
            # Fallback: return False for all skills
            return {skill: False for skill in company_skills}
            
    except Exception as e:
        print(f"Error in AI skills matching: {e}")
        # Fallback: return False for all skills
        return {skill: False for skill in company_skills}

def analyze_resume_skills(file_path: str, company_skills: List[str]) -> Dict[str, Any]:
    """
    Main function to analyze resume and match skills.
    
    Args:
        file_path (str): Path to resume file (PDF or image)
        company_skills (List[str]): List of skills required by company
        
    Returns:
        Dict[str, Any]: Complete analysis result
    """
    try:
        # Extract resume content
        resume_content = extract_resume_content(file_path)
        
        if not resume_content.strip():
            return {
                "error": "Could not extract content from resume",
                "skills_match": {skill: False for skill in company_skills},
                "total_skills": len(company_skills),
                "matched_skills": 0,
                "match_percentage": 0.0
            }
        
        # Match skills using AI
        skills_match = match_skills_with_ai(resume_content, company_skills)
        
        # Calculate statistics
        matched_skills = sum(1 for matched in skills_match.values() if matched)
        total_skills = len(company_skills)
        match_percentage = (matched_skills / total_skills * 100) if total_skills > 0 else 0.0
        
        return {
            "skills_match": skills_match,
            "total_skills": total_skills,
            "matched_skills": matched_skills,
            "match_percentage": round(match_percentage, 2),
            "resume_content_length": len(resume_content)
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "skills_match": {skill: False for skill in company_skills},
            "total_skills": len(company_skills),
            "matched_skills": 0,
            "match_percentage": 0.0
        }
