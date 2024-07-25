import streamlit as st
import os
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# 路径设置
tokenizer_path = "../BAAI_bge-m3"
gen_model_path = "../GLM-4-9B-Chat"

# 检查CUDA设备的可用性
assert torch.cuda.device_count() > 1, "至少需要两个CUDA设备"
assert torch.cuda.is_available(), "CUDA设备不可用"

# 设置设备
device_query = torch.device("cuda:1")  # 使用第二个可用的设备
device_gen = torch.device("cuda:0")    # 使用第一个可用的设备

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(tokenizer_path)

# 加载生成模型和tokenizer
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(gen_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)

# 确保模型加载正确后再移动到设备
model = model.to(device_query)
model = torch.nn.DataParallel(model, device_ids=[1])

gen_model = gen_model.to(device_gen)
gen_model = torch.nn.DataParallel(gen_model, device_ids=[0]).eval()

# 固定随机种子
torch.manual_seed(42)

# 加载FAISS索引
index_path = "../RAG/faiss_index/embedding.index"
index = faiss.read_index(index_path)

# 加载条目和文件名映射
entries = []
with open("../RAG/faiss_index/entries.txt", "r", encoding="utf-8") as f:
    for line in f:
        file_path, entry = line.strip().split('\t')
        # 去掉路径中的../reference
        file_path = file_path.replace("../reference_book/", "")
        entries.append((file_path, entry))

# 函数：进行检索
def search(query, top_k=5):
    # 对查询进行编码
    query_tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)["input_ids"].to(device_query)
    with torch.no_grad():
        query_embedding = model(query_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    # 检索最相似的top_k个结果
    distances, indices = index.search(query_embedding, top_k)
    results = [(entries[I], distances[0][j]) for j, I in enumerate(indices[0])]
    return results, query_embedding

# 函数：生成答案
def generate_answer(context, query):
    input_text = f"法律问题:{query}\n回答可能会用到的参考文献:{context}\n"
    inputs = gen_tokenizer(input_text, return_tensors="pt", truncation=True, max_length=gen_tokenizer.model_max_length).to(device_gen)
    gen_kwargs = {"max_length": 1024, "do_sample": True}  # 禁用采样

    with torch.no_grad():
        outputs = gen_model.module.generate(**inputs, **gen_kwargs)
        answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 去除input_text相关内容
    answer = answer.replace(input_text, "").strip()
    return answer

# RAG函数：检索并生成答案
def rag(query, top_k=5, similarity_threshold=0.9, max_results=3):
    search_results, query_embedding = search(query, top_k)
    
    # 去重
    unique_entries = []
    seen_embeddings = []
    seen_entries = []
    
    for (file_path, entry), distance in search_results:
        if len(unique_entries) >= max_results:
            break

        # 对条目进行编码
        entry_tokens = tokenizer(entry, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)["input_ids"].to(device_query)
        with torch.no_grad():
            entry_embedding = model(entry_tokens).last_hidden_state.mean(dim=1).cpu().numpy().reshape(1, -1)
        
        if seen_embeddings:
            seen_embeddings_array = np.vstack(seen_embeddings)
            similarities = cosine_similarity(entry_embedding, seen_embeddings_array).flatten()
            max_similarity_index = np.argmax(similarities)
            if max(similarities) < similarity_threshold:
                seen_embeddings.append(entry_embedding)
                seen_entries.append(entry)
                unique_entries.append((file_path, entry, distance))
            else:
                if len(entry) > len(seen_entries[max_similarity_index]):
                    seen_embeddings[max_similarity_index] = entry_embedding
                    seen_entries[max_similarity_index] = entry
                    unique_entries[max_similarity_index] = (file_path, entry, distance)
        else:
            seen_embeddings.append(entry_embedding)
            seen_entries.append(entry)
            unique_entries.append((file_path, entry, distance))
    
    context = " ".join([entry for _, entry, _ in unique_entries])
    answer = generate_answer(context, query)
    return answer, unique_entries

# Streamlit 界面
st.title("法律问题问答系统")

query = st.text_input("请输入您的法律问题：")

if query:
    with st.spinner('处理中...'):
        answer, results = rag(query, top_k=10, max_results=3)  # top_k 设为10以获取更多候选结果用于去重

    st.subheader("基于参考文献的回答:")
    st.write(answer)

    # 在侧边栏展示参考文献
    st.sidebar.subheader("参考文献:")
    for (filename, entry, distance) in results:
        st.sidebar.write(f"文件: {filename}\n条目: {entry.strip()}\n距离: {distance:.4f}")