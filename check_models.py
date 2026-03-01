from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

print("Available Models (support generateContent):")
for m in client.models.list():
    print(f"- {m.name}")
