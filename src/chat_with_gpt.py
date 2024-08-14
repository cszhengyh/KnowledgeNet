import requests
from openai import OpenAI

OPENAI_API_KEY = ""
client = OpenAI(api_key=OPENAI_API_KEY)
model = 'gpt-4o-2024-08-06'
headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {OPENAI_API_KEY}"
}

def chat_with_gpt(query):
    if isinstance(query, str):
        content = []
        messages = []        
        content.append({
            "type": "text",
            "text": query,
        })
        messages.append({'role': 'user', "content": content})
        query = messages

    while True:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=query
            )
            break
        except Exception as e:
            print("openai gpt4o request error.")
            
    resp_content = resp.choices[0].message.content
    return resp_content
