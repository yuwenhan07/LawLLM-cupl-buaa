import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json

# 路径设置
tokenizer_path = "/home/yuwenhan/tokenizer/BAAI_bge-m3"

# 加载tokenizer和模型
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(tokenizer_path)

# 设置设备为cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 使用多GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
model = model.to(device)

# 加载FAISS索引
index_path = "/home/yuwenhan/MVRAG/faiss_index/embedding2.index"
print("Loading FAISS index...")
index = faiss.read_index(index_path)

# 加载条目和文件名映射
entries = []
entries_path = "/home/yuwenhan/MVRAG/faiss_index/entries2.txt"
print(f"Loading entries from {entries_path}...")
with open(entries_path, "r", encoding="utf-8") as f:
    for line in f:
        file_path, entry = line.strip().split('\t')
        # 删除指定的路径部分
        file_path = file_path.replace("/home/yuwenhan/GLM-test-Lawbanch/JEC-QA/reference_book/", "")
        entries.append((file_path, entry))
print(f"Loaded {len(entries)} entries.")

# 函数：进行检索
def search(query, top_k=5):
    # 对查询进行编码
    query_tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)["input_ids"].to(device)
    with torch.no_grad():
        query_embedding = model(query_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    # 检索最相似的top_k个结果
    distances, indices = index.search(query_embedding, top_k)
    results = [entries[i] for i in indices[0]]
    return results

# 从JSON文件中读取查询
queries_path = '../data/xingfa.json'
print(f"Loading queries from {queries_path}...")
with open(queries_path, 'r', encoding='utf-8') as f:
    queries = json.load(f)
print(f"Loaded {len(queries)} queries.")

# 存储所有查询结果
all_results = []

for query_idx, query in enumerate(queries):
    query_text = query['text']
    print(f"Processing query {query_idx + 1}: {query_text}")
    query_results = search(query_text)
    
    # 结构化查询结果
    result_entry = {
        'query': query_text,
        'results': [
            {
                'index': result_idx + 1,
                'filename': filename,
                'entry': entry.strip()
            }
            for result_idx, (filename, entry) in enumerate(query_results)
        ]
    }
    all_results.append(result_entry)

# 将结果写入JSON文件
results_path = '../data/search_results.json'
print(f"Writing results to {results_path}...")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)
print("Finished writing results.")