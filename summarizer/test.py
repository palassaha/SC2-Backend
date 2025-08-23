import json
from typing import Dict, Optional, List, Any
import ollama
import re


SCHEMA_KEYS = [
    "company",
    "website",
    "registration_link",
    "role",
    "ctc",
    "type",
    "criteria",
    "responsibilities",
    "benefits",
    "applicationProcess",
    "eligibility",
    "content"  # Added new field
]

SYSTEM_PROMPT = """You are a precise information extraction assistant for job/internship postings.
You will be given job/internship text and must extract specific fields in strict JSON format.
ALWAYS return STRICT JSON with these exact keys (no extra keys, no explanations):

{
  "company": "",
  "website": "",
  "registration_link": "",
  "role": "",
  "ctc": "",
  "type": "",
  "criteria": {
    "cgpa": null,
    "backlogs": null,
    "skills": [],
    "courses": [],
    "experience": ""
  },
  "responsibilities": [],
  "benefits": [],
  "applicationProcess": [],
  "eligibility": {
    "minCGPA": "",
    "branches": [],
    "batch": []
  },
  "content": ""
}

IMPORTANT EXTRACTION RULES:
- Output must be ONLY that JSON object and nothing else.
- "company": Extract company/organization name from text
- "website": Company website URL if mentioned
- "registration_link": Application/registration/form link if provided
- "role": Job/internship position title or generic role if multiple positions
- "ctc": Full-time salary after internship OR current salary/stipend (include currency and period)
- "type": Must be one of: "Internship", "Job", "Announcement", "Opportunity", "Deadline", "Update"

CRITERIA EXTRACTION:
- "criteria.cgpa": Convert percentage requirements to CGPA (80% = 8.0, 85% = 8.5, etc.). Use 0 if "No backlogs" or similar restrictions mentioned
- "criteria.backlogs": 0 if "No backlogs" mentioned, otherwise null
- "criteria.skills": Technical skills mentioned (programming languages, frameworks, development areas)
- "criteria.courses": Relevant academic courses/subjects mentioned
- "criteria.experience": Experience level requirements

DETAILED EXTRACTION:
- "responsibilities": Extract job duties, work description, what the person will do. Look for job descriptions, work tasks, daily activities
- "benefits": All perks mentioned (stipend amount, certificates, mentorship, travel allowances, meals, accommodation, etc.)
- "applicationProcess": Step-by-step application process including deadlines and important dates
- "eligibility.minCGPA": Academic percentage/CGPA requirements as mentioned in text
- "eligibility.branches": Eligible academic branches/departments (CSE, ECE, IT, etc. or "All" if mentioned)
- "eligibility.batch": Graduation years/batches mentioned
- "content": Leave this empty - it will be processed separately

CONVERSION RULES:
- Convert percentage to CGPA: 80% = 8.0, 85% = 8.5, 75% = 7.5
- Extract ALL benefits mentioned including monetary and non-monetary
- Look carefully for job responsibilities in job descriptions or role details
- Use plain text; no markdown formatting.
"""


def safe_str(value: Optional[str]) -> str:
    """Ensure a value is always a stripped string."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    return value.strip()


def safe_list(value: Optional[List]) -> List:
    """Ensure a value is always a list."""
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return value


def safe_dict(value: Optional[Dict]) -> Dict:
    """Ensure a value is always a dictionary."""
    if value is None or not isinstance(value, dict):
        return {}
    return value


def _coerce_json_from_text(text: str) -> Dict[str, Any]:
    """
    Extract the JSON object from the LLM response.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to find the first {...} and parse
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
        else:
            raise ValueError("Could not parse JSON from LLM response.")


def _analyze_and_htmlize_content(raw_text: str) -> str:
    """
    Analyze the raw text, divide it into logical points, and format as HTML.
    """
    # Clean and normalize the text
    text = re.sub(r'\s+', ' ', raw_text.strip())
    
    # Split text into sentences and paragraphs
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences into logical sections
    sections = []
    current_section = []
    
    # Keywords that typically start new sections
    section_keywords = [
        'company', 'role', 'position', 'internship', 'job', 'requirements', 
        'eligibility', 'criteria', 'responsibilities', 'benefits', 'perks',
        'application', 'process', 'deadline', 'registration', 'stipend',
        'salary', 'ctc', 'location', 'duration', 'skills', 'qualifications'
    ]
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        # Check if this sentence starts a new section
        starts_new_section = False
        for keyword in section_keywords:
            if sentence_lower.startswith(keyword) or f' {keyword}' in sentence_lower[:20]:
                starts_new_section = True
                break
        
        if starts_new_section and current_section:
            sections.append(' '.join(current_section))
            current_section = [sentence]
        else:
            current_section.append(sentence)
    
    # Add the last section
    if current_section:
        sections.append(' '.join(current_section))
    
    # If we don't have good sections, fall back to paragraph splitting
    if len(sections) <= 2:
        paragraphs = re.split(r'\n\s*\n', raw_text.strip())
        sections = [p.strip() for p in paragraphs if p.strip()]
    
    # Convert to HTML
    html_content = "<div class='job-content'>\n"
    
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        # Detect section type based on content
        section_lower = section.lower()
        section_class = "content-point"
        
        if any(word in section_lower for word in ['company', 'organization']):
            section_class = "company-info"
        elif any(word in section_lower for word in ['role', 'position', 'job title']):
            section_class = "role-info"
        elif any(word in section_lower for word in ['requirement', 'criteria', 'eligibility']):
            section_class = "requirements"
        elif any(word in section_lower for word in ['benefit', 'perk', 'stipend', 'salary']):
            section_class = "benefits"
        elif any(word in section_lower for word in ['application', 'process', 'deadline']):
            section_class = "application-process"
        elif any(word in section_lower for word in ['responsibility', 'duties', 'work']):
            section_class = "responsibilities"
        
        # Format as HTML list item
        html_content += f"  <div class='{section_class}'>\n"
        html_content += f"    <p><strong>Point {i+1}:</strong> {section.strip()}</p>\n"
        html_content += f"  </div>\n"
    
    html_content += "</div>"
    
    return html_content


def _harden_schema(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all keys exist and have correct types."""
    result = {}
    
    # String fields
    for key in ["company", "website", "registration_link", "role", "ctc", "type"]:
        result[key] = safe_str(obj.get(key, ""))
    
    # Criteria object with better handling
    criteria = safe_dict(obj.get("criteria", {}))
    
    # Handle CGPA conversion from percentage
    cgpa_val = criteria.get("cgpa")
    if isinstance(cgpa_val, str) and "%" in cgpa_val:
        try:
            # Extract percentage and convert to CGPA
            percentage = float(cgpa_val.replace("%", "").strip())
            cgpa_val = percentage / 10.0  # 80% -> 8.0
        except:
            cgpa_val = None
    
    # Handle backlogs - if "no backlogs" mentioned, set to 0
    backlogs_val = criteria.get("backlogs")
    if isinstance(backlogs_val, str) and "no" in backlogs_val.lower():
        backlogs_val = 0
    
    result["criteria"] = {
        "cgpa": cgpa_val,
        "backlogs": backlogs_val,
        "skills": safe_list(criteria.get("skills", [])),
        "courses": safe_list(criteria.get("courses", [])),
        "experience": safe_str(criteria.get("experience", ""))
    }
    
    # Array fields
    for key in ["responsibilities", "benefits", "applicationProcess"]:
        result[key] = safe_list(obj.get(key, []))
    
    # Eligibility object
    eligibility = safe_dict(obj.get("eligibility", {}))
    result["eligibility"] = {
        "minCGPA": safe_str(eligibility.get("minCGPA", "")),
        "branches": safe_list(eligibility.get("branches", [])),
        "batch": safe_list(eligibility.get("batch", []))
    }
    
    # Content field - will be populated separately
    result["content"] = safe_str(obj.get("content", ""))
    
    return result


def extract_job_json(raw_text: str, model: str = "gemma3:latest", host: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract job information as structured JSON using Ollama LLM.

    Args:
        raw_text (str): The input text containing the job/drive description.
        model (str): Ollama model name.
        host (Optional[str]): Custom Ollama server host.

    Returns:
        Dict[str, Any]: Extracted job fields matching the Post schema with HTML content.
    """

    user_prompt = f"""Extract all required fields into JSON format ONLY, without any extra text.

CRITICAL INSTRUCTIONS:
- For "criteria.cgpa": Convert 80% to 8.0, 85% to 8.5, etc. If "No backlogs" mentioned, set to 0
- For "criteria.backlogs": Set to 0 if "No backlogs" is mentioned
- For "responsibilities": Look for job descriptions, work tasks, what interns/employees will do
- For "benefits": Include ALL benefits: stipend amounts, certificates, mentorship, travel, meals, stay, etc.
- For "eligibility.branches": Extract B.Tech branches or "All" if mentioned
- For "eligibility.batch": Extract graduation years like "2026"
- For "applicationProcess": Include registration steps and deadlines
- For "content": Leave this field empty - it will be processed separately

READ THE TEXT CAREFULLY and extract ALL mentioned information.

Input Job Description:
\"\"\"{raw_text.strip()}\"\"\"
"""

    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "options": {"temperature": 0.2}
    }

    # Call the Ollama LLM
    if host:
        client = ollama.Client(host=host)
        resp = client.chat(**kwargs)
    else:
        resp = ollama.chat(**kwargs)

    content = resp["message"]["content"]
    data = _coerce_json_from_text(content)
    result = _harden_schema(data)
    
    # Generate HTML content from the raw text
    result["content"] = _analyze_and_htmlize_content(raw_text)
    
    return result


def test_extraction(sample_text: str):
    """
    Extract job JSON from any text.
    """
    result = extract_job_json(sample_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


