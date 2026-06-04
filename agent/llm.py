import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# We use a placeholder for the API key if not present, but it will fail on actual calls if invalid.
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "invalid_key"))

def call_llm(prompt: str, system_prompt: str = "You are a helpful assistant.", temperature: float = 0.0, model: str = "llama-3.3-70b-versatile"):
    """
    Calls the Groq LLM and returns the content and token usage.
    """
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        model=model,
        temperature=temperature,
    )
    
    content = response.choices[0].message.content
    usage = response.usage
    return content, usage.prompt_tokens, usage.completion_tokens
