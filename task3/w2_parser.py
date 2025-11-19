import os
import json
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import google.generativeai as genai

# ----------- OCR Function -------------
def extract_text(file_path):
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=300)
        text = ""
        for p in pages:
            text += pytesseract.image_to_string(p)
        return text
    else:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img)

# ----------- Gemini Call --------------
def call_gemini(prompt, text):
    """
    Calls Gemini model to generate content.
    Ensures JSON output by appending explicit instruction.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    full_prompt = (
        prompt
        + "\n\nW2 TEXT:\n"
        + text
        + "\n\nRespond only in valid JSON format, suitable for parsing."
    )

    response = model.generate_content(full_prompt)
    result_text = getattr(response, "text", None)

    if not result_text:
        raise Exception("Gemini returned empty response")
    
    return result_text

# ----------- Mask helper --------------
def mask_last4(value):
    if value and value.isdigit() and len(value) >= 4:
        return "XXX-XX-" + value[-4:]
    return value

# ----------- Main Processor ------------
def process_w2(file_path, api_key):
    # Configure Gemini API key
    genai.configure(api_key=api_key)

    raw_text = extract_text(file_path)

    # Read prompts
    with open("prompts/extraction_prompt.txt") as f:
        extraction_prompt = f.read()

    # Generate JSON from Gemini
    json_str = call_gemini(extraction_prompt, raw_text)

    # Try parsing JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # If JSON fails, save raw response for debugging
        with open("debug_gemini_output.txt", "w", encoding="utf-8") as f:
            f.write(json_str)
        raise Exception(
            "Gemini returned invalid JSON. Raw output saved in debug_gemini_output.txt"
        )

    # Mask sensitive fields
    if "employee" in data:
        data["employee"]["ssn_last4"] = mask_last4(data["employee"]["ssn_last4"])
    if "employer" in data:
        data["employer"]["ein_last4"] = mask_last4(data["employer"]["ein_last4"])

    # Insights
    with open("prompts/insight_prompt.txt") as f:
        insight_prompt = f.read()

    insight_response = call_gemini(insight_prompt, json.dumps(data))
    
    try:
        insights = json.loads(insight_response)
    except json.JSONDecodeError:
        insights = {"raw_response": insight_response}

    return {
        "fields": data,
        "insights": insights,
        "quality": {
            "ocr_length": len(raw_text),
            "warnings": []
        }
    }
