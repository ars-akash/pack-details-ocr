from openai import OpenAI
from proc import input_frames

user_api = input("Enter Your API Key: \n")

client = OpenAI(api_key=user_api)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            
            "role": "user",
            "content": "Write a joke"
        }
    ]
)

print(completion.choices[0].message)
