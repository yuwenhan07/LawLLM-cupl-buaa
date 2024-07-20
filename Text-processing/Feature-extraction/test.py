import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained("../../GLM-4-9B-Chat", trust_remote_code=True)

# 输入查询
query = "被告人朱某某于2018年4月25日至4月27日期间，先后在常州市钟楼区**镇**东大门**路**号门口、**村委**村**号**楼**房间、**路**物流港**幢*号**楼一房间等地盗窃作案3次，窃得被害人吴某某、周某某、王某某的电动车、手机等物品。"

# 使用聊天模板对输入进行编码
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": query}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True
)

# 将输入移到设备上
inputs = inputs.to(device)

# 加载模型，并确保输出隐藏状态
model = AutoModelForCausalLM.from_pretrained(
    "../../GLM-4-9B-Chat",
    output_hidden_states=True,  # 确保输出隐藏状态
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

# 生成文本的参数
gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

# 无需梯度计算
with torch.no_grad():
    # 生成输出
    outputs = model.generate(**inputs, **gen_kwargs)
    # 去掉输入部分，保留生成的输出
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("生成的文本：", generated_text)

# 提取特征
with torch.no_grad():
    # 获取模型输出（包括隐藏状态）
    model_output = model(**inputs, output_hidden_states=True)
    hidden_states = model_output.hidden_states

# 获取最后一层的隐藏状态
last_hidden_state = hidden_states[-1]

# 计算整个句子的嵌入表示（例如，取平均值）
sentence_embedding = torch.mean(last_hidden_state, dim=1)

# 打印嵌入表示
print("句子级别的嵌入：", sentence_embedding)