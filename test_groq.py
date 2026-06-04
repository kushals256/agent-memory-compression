import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "hi"}],
        model="llama-3.3-70b-versatile",
    )
    print("Success")
except Exception as e:
    print(f"Error: {e}")
