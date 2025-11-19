# Prudent__Task

# Project Execution Guidelines

## Task 1 — Price Gap Pair Finder

I have created a file named `task1.py` where the function logic alone relies separately as modular programming, and for testing that I have created a separate file called `task1test.py`.

**To run the test:**
```bash
python task1test.py
```

## Task 2 — Web API for Pair Finder + Movie Lookup

For this task I have created a python file called `task2api.py`. This file makes use of the `task1.py` file for the POST API functionality.

**To execute this file:**
1. Install FastAPI:
```bash
   pip install fastapi uvicorn
```
2. Run the command:
```bash
   uvicorn task2api:app --reload
```

I have tested both APIs using Swagger.

## Task 3 — W-2 Parser & Insight Generator (Gemini)

**To execute this file:**
```bash
task3>python cli.py file "sample_w2.jpg" -out w2_result.json 
```

The input image file is also available inside this task folder itself which is `paystub.jpg`.

## Output

Every task's output screenshots are attached in the `output` folder.
