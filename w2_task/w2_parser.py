import hashlib
import os
import re
import json
import time
import argparse
from typing import Tuple, Dict, Any, List, Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import requests
from dotenv import load_dotenv

# Load environment
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
REQUEST_TIMEOUT = int(os.getenv("GEMINI_TIMEOUT", 60))
GEMINI_MODEL = "gemini-2.5-pro"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MAX_RETRIES = int(os.getenv("GEMINI_RETRIES", "1"))

print("Loaded Gemini API Key:", GEMINI_API_KEY[:12] + "...")
CACHE_DIR = "gemini_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Regex helpers
SSN_RE = re.compile(r'\b(\d{3})[- ]?(\d{2})[- ]?(\d{4})\b')
EIN_RE = re.compile(r'\b(\d{2})[- ]?(\d{7})\b')


def get_cache_path(prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.json")


def mask_ssn(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = SSN_RE.search(text)
    if m:
        return f"***-**-{m.group(3)}"
    last4 = re.search(r'(\d{4})\b', text)
    if last4:
        return f"***-**-{last4.group(1)}"
    return None


def mask_ein(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = EIN_RE.search(text)
    if m:
        return f"**-{m.group(2)}"
    last4 = re.search(r'(\d{4})\b', text)
    if last4:
        return f"****{last4.group(1)}"
    return None


def ocr_image(image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        raise RuntimeError(f"OCR failure: {e}")


def ocr_pdf(file_path: str) -> Tuple[List[str], List[Image.Image]]:
    try:
        pages = convert_from_path(file_path)
    except Exception as e:
        raise RuntimeError(f"PDF -> images conversion failed: {e}")
    texts = [ocr_image(p) for p in pages]
    return texts, pages


def ocr_file(file_path: str) -> Tuple[str, List[Image.Image], List[str], List[str]]:
    warnings, images, page_texts = [], [], []
    lower = file_path.lower()
    try:
        if lower.endswith(".pdf"):
            page_texts, images = ocr_pdf(file_path)
            combined = "\n\n".join(page_texts)
        else:
            img = Image.open(file_path).convert("RGB")
            images = [img]
            t = ocr_image(img)
            page_texts = [t]
            combined = t
    except Exception as e:
        warnings.append(f"OCR error: {e}")
        combined = ""
    return combined, list(images), list(page_texts), warnings


def call_gemini(prompt: str, max_tokens: int = 2000, temperature: float = 0.2, retries: int = 2, delay: float = 1.0):
    cache_file = get_cache_path(prompt)
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        print("Loaded from cache.")
        return cached["output"]

    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    def send_request(max_toks):
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_toks},
        }
        resp = requests.post(GEMINI_URL, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()

    for attempt in range(1, retries + 1):
        try:
            print(f"\nSending prompt ({len(prompt)} chars) to {GEMINI_MODEL} (attempt {attempt}) ...")
            data = send_request(max_tokens)

            candidate = data.get("candidates", [{}])[0]
            model_output = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
            finish_reason = candidate.get("finishReason", "")

            if not model_output.strip():
                print(f"Empty model output. Finish reason: {finish_reason}")
                if "MAX_TOKENS" in finish_reason and max_tokens < 4000:
                    print("Increasing max_tokens and retrying...")
                    time.sleep(delay)
                    max_tokens = min(max_tokens * 2, 4000)
                    continue
                raise RuntimeError("Empty model output — even after retry.")

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump({"output": model_output, "raw": data}, f, indent=2)
            print("Gemini response cached.")
            return model_output

        except Exception as e:
            print(f"Gemini call failed (attempt {attempt}): {e}")
            if attempt < retries:
                time.sleep(delay)
                continue
            raise RuntimeError(f"Gemini call failed after {attempt} attempts: {e}")


EXTRACTION_PROMPT = """
You are given the OCR text of a U.S. W-2 form. Return STRICTLY valid JSON...
OCR_TEXT:
```{OCR_TEXT}```
"""

INSIGHTS_PROMPT = """
You are given a normalized W-2 JSON. Return JSON with insights only:
{W2_JSON}
"""


def coerce_number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    v = str(value).strip().replace("$", "").replace(",", "")
    if v == "":
        return None
    if v.startswith("(") and v.endswith(")"):
        v = "-" + v[1:-1]
    try:
        return float(v)
    except:
        return None


def normalize_state(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = s.strip().upper()
    states = {
        "ALABAMA": "AL", "CALIFORNIA": "CA", "NEW YORK": "NY", "TEXAS": "TX",
        "FLORIDA": "FL", "ILLINOIS": "IL"
    }
    return states.get(s, s[:2])


def postprocess_fields(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    warnings = []
    fields = raw.copy()

    emp = fields.get("employee", {})
    emp["ssn"] = mask_ssn(emp.get("ssn"))
    emp2 = fields.get("employer", {})
    emp2["ein"] = mask_ein(emp2.get("ein"))

    for addr in (emp.get("address"), emp2.get("address")):
        if addr:
            addr["state"] = normalize_state(addr.get("state"))

    fed = fields.get("federal", {})
    for k, v in list(fed.items()):
        if k.startswith("box"):
            fed[k] = coerce_number(v)
            if fed[k] is None and v not in (None, ""):
                warnings.append(f"Could not parse {k} value: {v}")
    fields["federal"] = fed

    return fields, warnings


def process_w2(file_path: str, test_mode: bool = False) -> Dict[str, Any]:
    """
    Main entry. Returns dictionary with keys: fields, insights, quality.
    """
    import re, json

    quality = {"warnings": [], "confidence": None}

    # Step 1: OCR
    if test_mode:
        ocr_text = (
            "Employee: John Doe\nSSN: 123-45-6789\nEmployee address: 123 Main St Apt 4, Anytown, CA 90210\n"
            "Employer: ACME Corp\nEIN: 12-3456789\nBox1: 85000.00\nBox2: 7000.00\nBox3: 85000.00\nBox4: 5270.00\n"
        )
        images, page_texts = [], [ocr_text]
        quality["warnings"].append("TEST MODE: OCR bypassed.")
    else:
        ocr_text, images, page_texts, ocr_warnings = ocr_file(file_path)
        quality["warnings"].extend(ocr_warnings)
        if not ocr_text.strip():
            quality["warnings"].append("OCR produced empty text.")

    # Step 2: Extraction via Gemini
    try:
        if test_mode:
            extracted = {
                "employee": {"name": "John Doe", "ssn": "***-**-6789"},
                "employer": {"name": "ACME Corp", "ein": "**-56789"},
                "federal": {"box1": 85000, "box2": 7000},
                "confidence": 0.95,
                "parse_warnings": [],
            }
        else:
            ocr_text = ocr_text[:10000]
            prompt = EXTRACTION_PROMPT.replace("{OCR_TEXT}", ocr_text)
            raw_resp = call_gemini(prompt, max_tokens=4000)
            model_text = raw_resp if isinstance(raw_resp, str) else (
                raw_resp.get("text") or raw_resp.get("output") or ""
            )

            if not model_text.strip():
                raise RuntimeError("Gemini returned empty text. Check API key or prompt format.")

            # Try direct JSON parse
            try:
                extracted = json.loads(model_text)
            except Exception:
                print("Gemini output invalid JSON. Attempting repair...")

                # --- Safe JSON extraction without recursive regex ---
                def extract_first_json(text):
                    start = text.find('{')
                    if start == -1:
                        return None
                    depth = 0
                    for i, ch in enumerate(text[start:], start=start):
                        if ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                return text[start:i+1]
                    return None  # no balanced closing brace

                json_candidate = extract_first_json(model_text)
                if not json_candidate:
                    raise RuntimeError("Could not locate JSON object in Gemini output.")

                try:
                    extracted = json.loads(json_candidate)
                    quality["warnings"].append("Partial JSON extracted successfully.")
                except Exception as e2:
                    # try truncating last incomplete lines
                    clean_text = re.sub(r'[^}]*$', '}', json_candidate.strip())
                    extracted = json.loads(clean_text)
                    quality["warnings"].append("Cleaned truncated JSON successfully.")

    except Exception as e:
        raise RuntimeError(f"Extraction step failed: {e}")

    # Step 3: Postprocess (normalize, mask)
    normalized, post_warnings = postprocess_fields(extracted)
    quality["warnings"].extend(post_warnings)
    quality["confidence"] = extracted.get("confidence", None)

    # Step 4: Insights via Gemini
    try:
        if test_mode:
            insights_obj = {
                "insights": ["Sample test insights."],
                "confidence": 0.9,
                "notes": [],
            }
        else:
            insight_prompt = INSIGHTS_PROMPT.replace("{W2_JSON}", json.dumps(normalized))
            raw_resp2 = call_gemini(insight_prompt, max_tokens=800)
            # SAFER HANDLING: handle both dict or raw string response
            if isinstance(raw_resp2, dict):
                model_text2 = raw_resp2.get("text") or raw_resp2.get("output") or ""
            else:
                model_text2 = str(raw_resp2)


            if not model_text2.strip():
                raise RuntimeError("Gemini returned empty text for insights.")

            try:
                insights_obj = json.loads(model_text2)
            except Exception:
                # try to extract JSON block manually again
                json_fragment = extract_first_json(model_text2)
                if json_fragment:
                    insights_obj = json.loads(json_fragment)
                    quality["warnings"].append("Extracted insights JSON fragment.")
                else:
                    insights_obj = {
                        "insights": ["Could not parse Gemini insights output."],
                        "confidence": None,
                        "notes": [],
                    }
                    quality["warnings"].append("Insights parse error.")
    except Exception as e:
        raise RuntimeError(f"Insights step failed: {e}")

    # Step 5: Merge results
    quality["model_confidence_extraction"] = extracted.get("confidence")
    quality["model_confidence_insights"] = insights_obj.get("confidence")
    quality["warnings"].extend(extracted.get("parse_warnings") or [])

    return {
        "fields": normalized,
        "insights": insights_obj.get("insights", []),
        "quality": quality,
    }




def main():
    parser = argparse.ArgumentParser(description="Process a W-2 file and produce structured JSON + insights.")
    parser.add_argument("--file", "-f", required=True, help="Path to W-2 file (pdf/image)")
    parser.add_argument("--test", action="store_true", help="Test mode (no API calls)")
    parser.add_argument("--out", "-o", help="Write JSON result to file")
    args = parser.parse_args()

    result = process_w2(args.file, test_mode=args.test)
    out_json = json.dumps(result, indent=2)
    print(out_json)

    if args.out:
        with open(args.out, "w") as f:
            f.write(out_json)
        print(f"Wrote output to {args.out}")


if __name__ == "__main__":
    main()
