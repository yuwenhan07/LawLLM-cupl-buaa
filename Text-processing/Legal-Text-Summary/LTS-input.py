import os
import ast

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def predict(messages, model, tokenizer, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 输出去重函数
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for item in data:
        item_tuple = (item['label'], item['text'])
        if item_tuple not in seen:
            seen.add(item_tuple)
            unique_data.append(item)
    return unique_data

# Load tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("../../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

# Load the fine-tuned Lora model
model = PeftModel.from_pretrained(model, model_id="../../finetune/output//LTS/checkpoint-100")

# 确定使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the instruction
instruction = ("请对下面给的这篇法律文书提取摘要，用更短、更连贯、更自然的文字表达其主要内容。")

# Get user input
input_value = input("请输入需要进行摘要的法律文书: ")

# Prepare messages for the model
messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": f"{input_value}"}
]
# print('\n'+str(messages))

# Generate response
response = predict(messages, model, tokenizer, device)

print('原始文本：' + input_value + '\n' + '摘要后文本:' + response)
