import streamlit as st
import os
import faiss
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

# 路径设置
tokenizer_path = "../BAAI_bge-m3"
gen_model_path = "../GLM-4-9B-Chat"


# 设置设备
device_query = torch.device("cuda:1")  # 使用第二个可用的设备
device_gen = torch.device("cuda:0")    # 使用第一个可用的设备

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModel.from_pretrained(tokenizer_path)

# 加载生成模型和tokenizer
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_path, trust_remote_code=True)


# 确保模型加载正确后再移动到设备
model = model.to(device_query)
model = torch.nn.DataParallel(model, device_ids=[1])





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



# 函数：生成答案
def generate_answer(context, query):
    input_text = f"法律问题:{query}\n回答可能会用到的参考文献:{context}"
    inputs = gen_tokenizer.apply_chat_template([{"role": "user", "content": input_text}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )
    inputs = inputs.to(device_gen)
    gen_model = AutoModelForCausalLM.from_pretrained(
    gen_model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
    ).to(device_gen).eval()
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

    with torch.no_grad():
        answer = gen_model.generate(**inputs, **gen_kwargs)
        answer = answer[:, inputs['input_ids'].shape[1]:]
        answer = gen_tokenizer.decode(answer[0], skip_special_tokens=True)
    # 去除input_text相关内容
    # answer = answer.replace(input_text, "").strip()
    return answer

# 函数：进行检索
def search(query, top_k):
    # 对查询进行编码
    query_tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)["input_ids"].to(device_query)
    with torch.no_grad():
        query_embedding = model(query_tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    # 检索最相似的top_k个结果
    distances, indices = index.search(query_embedding, top_k)
    results = [(entries[I][0], entries[I][1]) for j, I in enumerate(indices[0])]
    
    # 去重和过滤包含关系的条目
    filtered_results = []
    seen_entries = set()
    
    for filename, entry in results:
        if any(entry in e for e in seen_entries):
            continue
        filtered_results.append((filename, entry))
        seen_entries.add(entry)
        
    # 将过滤后的结果格式化为参考文献文本
    context = "\n".join([f"{entry}" for _, entry in filtered_results])
    response = generate_answer(context, query)
    
    return response, filtered_results

# Streamlit 界面
st.title("智能法律问答系统")


# 示例查询
query = st.text_input("请输入您的法律问题：")
if query:
    with st.spinner('处理中...'):
        answer, results = search(query, top_k=3)
    st.subheader("基于参考文献的回答:")
    st.write(answer)
    # 在侧边栏展示参考文献
    st.sidebar.subheader("参考文献:")
    for (filename, entry) in results:
        st.sidebar.write(f"文件: {filename} \n 条目: {entry.strip()}")
