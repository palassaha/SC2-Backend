import json
from typing import Dict, Optional, List, Any
import ollama


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
    "eligibility"
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
  }
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
    
    return result


def extract_job_json(raw_text: str, model: str = "gemma3:latest", host: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract job information as structured JSON using Ollama LLM.

    Args:
        raw_text (str): The input text containing the job/drive description.
        model (str): Ollama model name.
        host (Optional[str]): Custom Ollama server host.

    Returns:
        Dict[str, Any]: Extracted job fields matching the Post schema.
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
    return _harden_schema(data)

def test_extraction(sample_text: str):
    """
    Extract job JSON from any text.
    """
    result = extract_job_json(sample_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result