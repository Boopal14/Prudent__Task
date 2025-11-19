import google.generativeai as genai

# Configure your API key
genai.configure(api_key="AIzaSyDpADi8Yq2yH9EcQzw6CmqB4kTQImjwlGI")

# List available models
models = genai.list_models()

for m in models:
    print(m.name)
