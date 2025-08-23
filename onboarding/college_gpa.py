from groq import Groq
import base64
import os
from dotenv import load_dotenv
import json

load_dotenv()

def encode_image(image_path: str) -> str:
    """Convert an image file into base64 string for API input."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_gpa_from_image(image_path: str) -> str:
    """
    Extract GPA from a college marksheet IMAGE using Groq Vision model.
    Args:
        image_path (str): Path to the marksheet image (jpg/png)
    Returns:
        str: Extracted GPA (if found), returns "0" if GPA is "XP", else a message
    """
    
    # --- Step 1: Encode the image ---
    base64_image = encode_image(image_path)
    
    # --- Step 2: Initialize Groq client ---
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    # --- Step 3: Build prompt and send request ---
    prompt = """
    You are an information extractor.
    Look at this marksheet image and extract ONLY the GPA.
    If GPA is not found, return: {"GPA": None}
    If GPA shows "XP" or similar fail/expelled status, return: {"GPA": "0"}
    Respond strictly in JSON format like: {"GPA": "9.1"} or {"GPA": "0"}
    """
    
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
    
    # --- Step 4: Parse response ---
    content = chat_completion.choices[0].message.content.strip()
    print("Raw Content:", content)
    
    try:
        data = json.loads(content)
        gpa = data.get("GPA", None)

        return gpa
    except json.JSONDecodeError:
        return "Parsing Error"