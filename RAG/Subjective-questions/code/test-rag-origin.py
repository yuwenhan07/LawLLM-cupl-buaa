import os
# 指定使用的 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("/home/yuwenhan/model/GLM-4-9B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "/home/yuwenhan/model/GLM-4-9B-Chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

# 读取输入 JSON 文件
with open('../data/xingfa.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 读取检索结果 JSON 文件
with open('../data/search_results.json', 'r', encoding='utf-8') as f:
    search_results = json.load(f)

responses = []

# 创建一个字典来快速查找检索结果
search_results_dict = {entry['query']: entry['results'] for entry in search_results}

for entry in data:
    entry_id = entry['id']
    text = entry['text']
    
    response_entry = {'id': entry_id, 'responses': {}}

    for i in range(1, 6):  # 假设最多有5个问题
        question_key = f'question{i}'
        if question_key in entry:
            question = entry[question_key]
            query = "以下是一道国家统一法律职业资格考试的主观题，请你从法学生的角度作答 " + f"{text} {question}" + " \n以下是一些你可能会用到的参考文献:\n"

            # 查找检索结果
            retrieved_docs = search_results_dict.get(query, [])[:3]  # 只取前3条

            # 合并检索到的文档内容作为上下文
            context = " ".join([doc['entry'] for doc in retrieved_docs])

            # 准备输入数据
            query_with_context = f"{query} {context}"
            inputs = tokenizer.apply_chat_template([{"role": "user", "content": query_with_context}],
                                                   add_generation_prompt=True,
                                                   tokenize=True,
                                                   return_tensors="pt",
                                                   return_dict=True
                                                   )
            inputs = inputs.to(device)
            
            # 生成回答
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response_entry['responses'][question_key] = response
    
    responses.append(response_entry)

# 输出到 JSON 文件
with open('../data/output-rag-origin.json', 'w', encoding='utf-8') as f:
    json.dump(responses, f, ensure_ascii=False, indent=4)