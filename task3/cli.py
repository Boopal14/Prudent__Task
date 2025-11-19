import argparse
from w2_parser import process_w2
import json
import os

# Hardcoded API key for testing
GEMINI_API_KEY = "AIzaSyDpADi8Yq2yH9EcQzw6CmqB4kTQImjwlGI"
print("Loaded GEMINI_API_KEY:", GEMINI_API_KEY)

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="Path to W2 file")
parser.add_argument("--out", default="result.json")

args = parser.parse_args()

result = process_w2(args.file, GEMINI_API_KEY)

with open(args.out, "w") as f:
    json.dump(result, f, indent=4)

print("Saved →", args.out)
