import os
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

# 路径设置
tokenizer_path = "../../BAAI_bge-m3"
base_dir = "../reference_book"

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

# 分块大小和重叠设置
chunk_size = min(512, tokenizer.model_max_length)  # 确保分块大小不超过模型的最大序列长度
overlap = 50

# 函数：读取文件内容并分成条目（例如每一行或每一段）
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().splitlines()  # 这里假设每一行是一个条目

# 函数：分块文本
def chunk_text(text, chunk_size, overlap):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end].tolist())
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

# 函数：编码文本
def encode_chunks(chunks):
    embeddings = []
    for chunk in chunks:
        inputs = torch.tensor(chunk).unsqueeze(0).to(device)  # Add batch dimension and move to GPU
        with torch.no_grad():
            embedding = model(inputs).last_hidden_state.mean(dim=1)
        embeddings.append(embedding.cpu().numpy())
    return np.vstack(embeddings)

# 函数：遍历目录并编码文本条目
def create_embeddings(directory, chunk_size, overlap):
    embeddings = []
    entries = []
    # 获取文件数量
    total_files = sum([len(files) for _, _, files in os.walk(directory)])
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    file_entries = read_file(file_path)
                    for entry in file_entries:
                        chunks = chunk_text(entry, chunk_size, overlap)
                        entry_embeddings = encode_chunks(chunks)
                        embeddings.append(entry_embeddings)
                        entries.extend([(file_path, entry)] * entry_embeddings.shape[0])
                    pbar.update(1)
    return np.vstack(embeddings), entries

# 函数：创建并保存FAISS索引
def create_faiss_index(embeddings, index_path, use_gpu=True):
    # 确保目录存在
    directory_path = os.path.dirname(index_path)
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    dim = embeddings.shape[1]
    if use_gpu and torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, dim)
        index_cpu = faiss.index_gpu_to_cpu(index)  # 将GPU索引转换为CPU索引
    else:
        index = faiss.IndexFlatL2(dim)
        index_cpu = index
    index_cpu.add(embeddings)
    faiss.write_index(index_cpu, index_path)

# 主程序
if __name__ == "__main__":
    embeddings, entries = create_embeddings(base_dir, chunk_size, overlap)
    index_path = "../faiss_index/embedding.index"
    create_faiss_index(embeddings, index_path, use_gpu=True)
    
    # 保存条目和文件名映射
    with open("../faiss_index/entries.txt", "w", encoding="utf-8") as f:
        for file_path, entry in entries:
            f.write(f"{file_path}\t{entry}\n")
