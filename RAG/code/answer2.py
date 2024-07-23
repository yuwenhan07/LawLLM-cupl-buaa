import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"

import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 路径设置
tokenizer_path = "../../BAAI_bge-m3"
gen_model_path = "../../GLM-4-9B-Chat"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(tokenizer_path)

# 加载生成模型和tokenizer
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

# 设置设备为cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
gen_model = gen_model.to(device).eval()

# 固定随机种子
torch.manual_seed(42)

# 加载FAISS索引
index_path = "../faiss_index/embedding.index"
index = faiss.read_index(index_path)

# 加载条目和文件名映射
entries = []
with open("../faiss_index/entries.txt", "r", encoding="utf-8") as f:
    for line in f:
        file_path, entry = line.strip().split('\t')
        entries.append((file_path, entry))

# 函数：生成答案
def generate_answer(context, query):
    input_text = f"法律问题:{query}\n回答可能会用到的参考文献:{context}\n"
    inputs = gen_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=gen_tokenizer.model_max_length).to(device)
    gen_kwargs = {"max_length": 1024, "do_sample": True}

    with torch.no_grad():
        outputs = gen_model.generate(**inputs, **gen_kwargs)
        answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去除input_text相关内容
    answer = answer.replace(input_text, "").strip()
    return answer

# 函数：进行检索
def search(query, top_k=5):
    # 对查询进行编码
    query_tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)["input_ids"].to(device)
    with torch.no_grad():
        query_embedding = model(query_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    # 检索最相似的top_k个结果
    distances, indices = index.search(query_embedding, top_k)
    results = [(entries[I][0], entries[I][1], distances[0][j]) for j, I in enumerate(indices[0])]
    
    # 去重和过滤包含关系的条目
    filtered_results = []
    seen_entries = set()
    
    for filename, entry, distance in results:
        if any(entry in e for e in seen_entries):
            continue
        filtered_results.append((filename, entry, distance))
        seen_entries.add(entry)
        
    # 将过滤后的结果格式化为参考文献文本
    context = "\n".join([f"{entry}" for _, entry, _ in filtered_results])
    response = generate_answer(context, query)
    
    return response, filtered_results

# 示例查询
query = input("请输入您的法律问题：")
answer, results = search(query, top_k=3)  

print(f"基于参考文献的回答: {answer}")
print("参考文献:")
for (filename, entry, distance) in results:
    print(f"文件: {filename}, 条目: {entry.strip()}, 距离: {distance}")