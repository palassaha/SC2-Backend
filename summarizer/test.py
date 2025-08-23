# ollama_job_extractor.py
# pip install ollama
# Make sure Ollama server is running: ollama serve
# Pull your gemma3n model: ollama pull gemma3n

import json
from typing import Dict, Optional
import ollama


SCHEMA_KEYS = [
    "company_name",
    "job_title",
    "job_description",
    "summarization",
    "job_role",
    "duration",
    "skill requirements",
    "criteria",
    "ctc",
]

SYSTEM_PROMPT = """You are a precise information extraction assistant.
You will be given a block of job/drive text. Extract the fields requested by the user.
If something is not explicitly present, infer carefully from context; otherwise return an empty string.
ALWAYS return STRICT JSON with these exact keys (no extra keys, no explanations):

{
  "company_name": "",
  "job_title": "",
  "job_description": "",
  "summarization": "",
  "job_role": "",
  "duration": "",
  "skill requirements": "",
  "criteria": "",
  "ctc": ""
}

Rules:
- Output must be ONLY that JSON object and nothing else.
- Keep "job_title" and "job_role" concise; if multiple roles exist, use slash/comma separated string.
- "summarization" should be 2–3 sentence plain-language summary.
- "ctc" should capture annual CTC and stipend separately if present.
- "duration" should capture internship/contract period and any date bounds.
- "skill requirements" should list ONLY technical programming/development skills (programming languages, frameworks, libraries, databases, development tools, software technologies). Exclude soft skills, hardware requirements (webcam, internet), infrastructure needs, and general job requirements.
- "criteria" should include academic requirements like: 10th score, 12th score, graduation percentage, minimum GPA, eligible departments/branches, no backlogs policy, and other eligibility conditions.
- Use plain text; no markdown.
"""


def safe_str(value: Optional[str]) -> str:
    """Ensure a value is always a stripped string."""
    if value is None:
        return ""
    if not isinstance(value, str):
        return str(value)
    return value.strip()


def _coerce_json_from_text(text: str) -> Dict[str, str]:
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


def _harden_schema(obj: Dict[str, str]) -> Dict[str, str]:
    """Ensure all keys exist and are strings."""
    return {k: safe_str(obj.get(k, "")) for k in SCHEMA_KEYS}


def extract_job_json(raw_text: str, model: str = "gemma3:latest", host: Optional[str] = None) -> Dict[str, str]:
    """
    Extract job information as structured JSON using Ollama LLM.

    Args:
        raw_text (str): The input text containing the job/drive description.
        model (str): Ollama model name.
        host (Optional[str]): Custom Ollama server host.

    Returns:
        Dict[str, str]: Extracted job fields in strict JSON format.
    """

    user_prompt = f"""Extract all required fields into JSON format ONLY, without any extra text.

For "skill requirements": Extract only technical programming/development skills like programming languages (Python, Java, JavaScript), frameworks (React, Django, Spring), libraries, databases (MySQL, MongoDB), and software development tools. DO NOT include hardware requirements (webcam, internet connectivity), infrastructure needs, or general job requirements.
For "criteria": Extract academic requirements like 10th/12th scores, graduation percentage, GPA requirements, eligible departments, backlogs policy, and other eligibility conditions.

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


# ------------------ EXAMPLE USAGE ------------------
def test_extraction(sample_text: str):
    """
    Extract job JSON from any text.
    """
    result = extract_job_json(sample_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

