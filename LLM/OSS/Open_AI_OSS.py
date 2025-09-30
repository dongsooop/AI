from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama OpenAI 호환
    api_key="ollama"                       # 임의 값
)

messages = [
    {"role":"system","content":"Reasoning: medium\nYou are a helpful assistant."},
    {"role":"user","content":"홍대입구역 10만원 데이트 코스를 표로 정리해줘."},
]

resp = client.chat.completions.create(
    model="gpt-oss:20b",
    messages=messages,
    temperature=0.7,
    max_tokens=256,
)
print(resp.choices[0].message.content)