import ollama
import json
from pathlib import Path

def extract_gpa_from_image(image_path: str, model: str = "granite3.2-vision:latest") -> str:
    """
    Extract GPA from a college marksheet IMAGE using Ollama Granite Vision model.
    Args:
        image_path (str): Path to the marksheet image (jpg/png)
        model (str): Ollama model name (default granite3.2-vision:latest)
    Returns:
        str: Extracted GPA (if found), else a message
    """

    # --- Step 1: Build the prompt ---
    prompt = """
    You are an information extractor.
    Look at this marksheet image and extract ONLY the GPA.
    If GPA is not found, return "NOT FOUND".
    Respond strictly in JSON format: {"GPA": "value"}
    """

    # --- Step 2: Run Ollama with image input ---
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt, "images": [str(Path(image_path).resolve())]}
        ]
    )

    content = response["message"]["content"].strip()

    # --- Step 3: Parse JSON response ---
    try:
        data = json.loads(content)
        return data.get("GPA", "NOT FOUND")
    except json.JSONDecodeError:
        return "Parsing Error"



