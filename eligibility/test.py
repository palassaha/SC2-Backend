import json
import ollama
from typing import Dict, Any

def extract_threshold(data: Dict[str, Any], model: str = "gemma3:latest") -> float:
    """
    Use LLM to extract numeric threshold percentage from eligibility criteria.
    """
    prompt = f"""
Extract the numeric percentage threshold from the following eligibility criteria:
"{data['eligibility_criteria']}"
Only return the number (e.g., 70).
"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    text = response["message"]["content"].strip()
    try:
        # Try to extract the first float/int in the response
        import re
        match = re.search(r"\d+(\.\d+)?", text)
        if match:
            return float(match.group())
    except Exception:
        pass
    # fallback threshold if extraction fails
    return 0.0

def check_eligibility(data: Dict[str, Any], model: str = "gemma3:latest") -> Dict[str, Any]:
    threshold = extract_threshold(data, model)

    # Python if-else checks for academic scores
    if data['class_10'] < threshold:
        return {"eligibility": "no", "reason": f"Class 10 score {data['class_10']}% is below threshold {threshold}%."}
    if data['class_12'] < threshold:
        return {"eligibility": "no", "reason": f"Class 12 score {data['class_12']}% is below threshold {threshold}%."}
    if data['gpa'] * 10 < threshold:
        return {"eligibility": "no", "reason": f"College score {data['gpa']} is below threshold {threshold}%."}

    # Remaining criteria checked by LLM
    prompt = f"""
You are an eligibility checking agent for campus placements.
Candidate details:
- Department: {data['department']}
- Active backlogs: {data['active_backlogs']}

Eligibility criteria: {data['eligibility_criteria']}

Rules:
1. Check if candidate's department matches the eligibility criteria.
2. Check if candidate has no active backlogs if mentioned.
3. Ignore academic scores; they have already been checked in Python.
Answer STRICTLY in this JSON format:
{{"eligibility": "yes" or "no", "reason": "explain which criteria passed or failed"}}
"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    try:
        return json.loads(response["message"]["content"])
    except Exception:
        text = response["message"]["content"]
        start = text.find("{")
        end = text.rfind("}") + 1
        return json.loads(text[start:end])

# ---------- Static Input ----------
input_data = {
    "eligibility_criteria": "Minimum 70% in academic score, no active backlogs, and only ECE department eligible",
    "department": "CSE",
    "class_10": 75,
    "class_12": 82,
    "gpa": 7.52,
    "active_backlogs": "no"
}

result = check_eligibility(input_data)
print(json.dumps(result, indent=2))
