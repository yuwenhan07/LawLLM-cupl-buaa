import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import ast

os.environ["CUDA_VISIBLE_DEVICES"] = "8"
CUDA_VISIBLE_DEVICES="8"

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
tokenizer = AutoTokenizer.from_pretrained("/home/yuwenhan/law-LLM/buaa&zgzf/finetune/ZhipuAI/glm-4-9b-chat/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/yuwenhan/law-LLM/buaa&zgzf/finetune/ZhipuAI/glm-4-9b-chat/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

# Load the fine-tuned Lora model
model = PeftModel.from_pretrained(model, model_id="/home/yuwenhan/law-LLM/buaa&zgzf/finetune/output/GLM4-NER-2/checkpoint-200")

# 确定使用的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the instruction
instruction = ("你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、"
               "盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果："
               "[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]")

# Get user input
input_value = input("请输入文本内容: ")

# Prepare messages for the model
messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": f"{input_value}"}
]

# Generate response
response = predict(messages, model, tokenizer, device)

# 使用 ast.literal_eval 将字符串转换为列表
data_list = ast.literal_eval(response)
unique_data = remove_duplicates(data_list)
print(unique_data)

import pandas as pd
# Convert the output to a table
def convert_to_table(response):
    df = pd.DataFrame([response])
    return df

# Use the function to convert response to table
df = convert_to_table(unique_data)
if df is not None:
    print(df)