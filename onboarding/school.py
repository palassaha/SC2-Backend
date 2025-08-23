from groq import Groq
import base64
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Encode local image into base64 string
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to calculate percentage from marks
def overall_percentage(marks, full_marks):
    obt = sum(marks)
    total = sum(full_marks)
    pct = obt / total * 100
    return f"{pct:.2f}%"


# Extract marks array from a marksheet image
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
                            # "show explanation"
                            # "Example: [85, 80, 75, 90]"
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

    # Try parsing JSON response
    content = chat_completion.choices[0].message.content.strip()
    try:
        marks = json.loads(content)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON: {content}")

    return marks


image_path = r"test_data\dibs_12.png"

marks = extract_marks_from_marksheet(image_path)
full = [100] * len(marks)

print("Extracted Marks:", marks)
print("Overall Percentage:", overall_percentage(marks, full))
