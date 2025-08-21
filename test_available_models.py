# This code is used to fetch the available models in the Groq:

import os, requests
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
url = "https://api.groq.com/openai/v1/models"
response = requests.get(url, headers=headers)
print(response.json())
