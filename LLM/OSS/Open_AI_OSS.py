import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

model_id = "unsloth/gpt-oss-20b-bnb-4bit"  # 4bit 사전양자화 체크포인트
tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=bnb)

pipe = pipeline("text-generation", model=model, tokenizer=tok)
msgs = [{"role":"system","content":"Reasoning: medium"}, {"role":"user","content":"테스트"}]
print(pipe(msgs, max_new_tokens=128)[0]["generated_text"][-1]["content"])