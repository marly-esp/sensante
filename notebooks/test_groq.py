import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

for temp in [0.0, 0.5, 1.0]:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Tu es un assistant médical sénégalais. Maximum 2 phrases."},
            {"role": "user", "content": "Patient : F, 28 ans, Dakar. Diagnostic : paludisme 72%. Explique."}
        ],
        max_tokens=150,
        temperature=temp
    )
    print(f"\n=== Temperature {temp} ===")
    print(response.choices[0].message.content)