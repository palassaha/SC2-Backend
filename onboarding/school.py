
from groq import Groq
import base64
import os
from dotenv import load_dotenv
import json
import re

load_dotenv()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def overall_percentage(marks, full_marks):
    obt = sum(marks)
    total = sum(full_marks)
    pct = obt / total * 100
    return f"{pct:.2f}%"

def extract_json_array(content: str):
    
    code_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
    match = re.search(code_block_pattern, content, re.DOTALL)
    if match:
        return json.loads(match.group(1))

    json_pattern = r'\[(?:\s*\d+(?:\s*,\s*\d+)*\s*)?\]'
    match = re.search(json_pattern, content)
    if match:
        return json.loads(match.group(0))
  
    return json.loads(content)


def extract_marks_from_marksheet(image_path: str):
    base64_image = encode_image(image_path)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    chat_completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are given a school marksheet image. "
                            "Return ONLY a JSON array of integers representing the obtained marks "
                            "in compulsory subjects (exclude optional/elective). "
                            "Do not return percentage, explanations, or extra text. "
                            "Example format: [85, 80, 75, 90]"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    content = chat_completion.choices[0].message.content.strip()
    try:
        print("Raw Content:", content)
        marks = extract_json_array(content)
        print("Extracted Marks:", marks)

        if not isinstance(marks, list) or not all(isinstance(x, int) for x in marks):
            raise ValueError("Extracted data is not a list of integers")
            
        full = [100] * len(marks)
        pct = overall_percentage(marks, full)
        print("Overall Percentage:", pct)
        return pct
        
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing response: {e}")
        raise ValueError(f"Model did not return valid JSON array: {content}")