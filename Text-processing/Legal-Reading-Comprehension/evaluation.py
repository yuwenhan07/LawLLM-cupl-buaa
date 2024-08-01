import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm  # 导入tqdm库

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("../../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="../../finetune/output/LRC/checkpoint-500")

# 读取测试数据 (JSONL)
test_texts = []
with open("./data/test.jsonl", "r") as f:
    for line in f:
        test_texts.append(json.loads(line))

# 生成回答
responses = []
for item in tqdm(test_texts, desc="Generating Responses"):  # 使用tqdm显示进度条
    instruction = "请你根据下面提供的'法律文本材料'内容，回答相应的'问题'，以完成片段抽取式的阅读理解任务。\n具体来说，你要正确回答'问题'，并且答案限定是'法律文本材料'的一个子句（或片段）。请你以'''答案：A'''的格式给出回答，其中A表示'法律文本材料'中正确的子句（或片段）。"
    input_value = item['text']

    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    responses.append({"input": input_value, "response": response})

# 保存回答 (JSONL)
with open("./data/model_responses.jsonl", "w", encoding="utf-8") as f:
    for response in responses:
        f.write(json.dumps(response, ensure_ascii=False) + "\n")
# 计算正确率
correct_count = 0
total_count = len(test_texts)
for item, response in zip(test_texts, responses):
    expected_output = item.get('answer', '').strip()
    actual_output = response['response'].strip()
    if expected_output == actual_output:
        correct_count += 1

accuracy = correct_count / total_count
print(f"Correctness: {accuracy:.2f}")