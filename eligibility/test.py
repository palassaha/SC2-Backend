import json
import ollama
from typing import Dict, Any

def build_prompt(data: Dict[str, Any]) -> str:
    return f"""
You are an eligibility checking agent for campus placements.

Eligibility criteria: {data['eligibility_criteria']}
Candidate details:
- Department: {data['department']}
- Class 10 score: {data['class_10']}%
- Class 12 score: {data['class_12']}%
- GPA: {data['gpa']}
- Active backlogs: {data['active_backlogs']}

Rules:
- If eligibility mentions "80% in academic score", it means Class 10 ≥ 80, Class 12 ≥ 80, and GPA ≥ 8.0.
- GPA is on a 10-point scale.
- The department of the student must also match the eligibility criteria.
- If candidate fails any required condition, mark them as "no".
- If candidate passes all required conditions, mark them as "yes".

Answer STRICTLY in this JSON format only:
{{"eligibility": "yes" or "no", "reason": "explain briefly"}}
"""

def check_eligibility(data: Dict[str, Any], model: str = "gemma:2b") -> Dict[str, Any]:
    prompt = build_prompt(data)
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    try:
        return json.loads(response["message"]["content"])
    except Exception:
        text = response["message"]["content"]
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])


# ---------- Static Input + Direct Call ----------
input_data = {
    "eligibility_criteria": "Minimum 80% in academic score, no active backlogs, and only CSE department eligible",
    "department": "CSE",
    "class_10": 85,
    "class_12": 82,
    "gpa": 8.5,
    "active_backlogs": "no"
}

result = check_eligibility(input_data)
print(json.dumps(result, indent=2))
