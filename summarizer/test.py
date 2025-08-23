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


text = '''Dear PR,

Greetings!

This is to inform you that RCC Institute of Information Technology is going to organize a campus drive by "OneBanc ” for the 2026 passing out batch B.Tech (All) & MCA

Find below the drive details:
The students will be able to join immediately as Interns for 1 year of Internship
Students who apply will be comfortable relocating to Gurugram, Haryana and working 6 days from office.
The students will require 20 to 25 days of leaves for their semester exam and final year exams.
You can find the details of the opportunity as mentioned below:



Skill

Job Description

Android Development

Here

iOS Development

Here

Full Stack Development

Here

UI/UX Design

Here

Data Science

Here

 

Here are the stipend details:

Stipend

Amount

Payout

Fixed

INR 40,000

Paid Monthly

Total

INR 40,000

NA

 

Note: The Annual CTC offered for the above-mentioned profiles after the completion of internship will be 12 LPA, based on performance. The duration of Internship will be till 30th June 2026 or last day of the month in which they get their final year results, whichever is later.

About Company:
OneBanc, is a neo-bank, building the economic infrastructure for the workforce of India. The idea of OneBanc started when a young girl asked Vibhore, a serial entrepreneur, why the money in her piggybank never grew. Adopting this philosophy of #DemandMore, OneBanc connects enterprises, banks, and HR Tech platforms to enhance value for all stakeholders. The core team has proven their vision and executive prowess in CoCubes – a complete assessment solution for students and institutes, which was acquired by Aon. They are now building the squad to enable the FinTech revolution of the future. 


Eligibility Criteria:

80% & above in 10th, 12th and Graduation
No backlogs
Here are some things to take note of, before candidates apply to OneBanc: 

This is a full-time opportunity based in Gurugram.
Relocation to Gurugram is a must and so is working from office.  
OneBanc takes care of candidates’ travel, meals and stay during the Pre-Hiring Evaluation.
Registration Link for eligible unplaced students: https://docs.google.com/forms/d/e/1FAIpQLSfEIAyPnfSpHfRMKKj4ywPhEpv1LcNtcTB3i1h8aW2qL9dLMQ/viewform?usp=pp_url
Last date for registration: 19.08.2025 (10.30 am)'''


job_data = extract_job_json(text)
print(job_data)