import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# 路径设置
tokenizer_path = "../../BAAI_bge-m3"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(tokenizer_path)

# 设置设备为cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 使用多GPU
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model = model.to(device)

# 加载FAISS索引
index_path = "../faiss_index/embedding.index"
index = faiss.read_index(index_path)

# 加载条目和文件名映射
entries = []
with open("../faiss_index/entries.txt", "r", encoding="utf-8") as f:
    for line in f:
        file_path, entry = line.strip().split('\t')
        entries.append((file_path, entry))

# 函数：进行检索
def search(query, top_k=5):
    # 对查询进行编码
    query_tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)["input_ids"].to(device)
    with torch.no_grad():
        query_embedding = model(query_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    # 检索最相似的top_k个结果
    distances, indices = index.search(query_embedding, top_k)
    results = [(entries[i], distances[0][j]) for j, i in enumerate(indices[0])]
    return results

# 示例查询
query = "我国立法法规定的立法原则为"
results = search(query)

for (filename, entry), distance in results:
    print(f"文件: {filename}, 条目: {entry.strip()}, 距离: {distance}")
