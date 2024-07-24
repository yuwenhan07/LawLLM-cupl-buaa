import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# Define predict function
def predict(messages, model, tokenizer, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# Output deduplication function
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for item in data:
        item_tuple = (item['label'], item['text'])
        if item_tuple not in seen:
            seen.add(item_tuple)
            unique_data.append(item)
    return unique_data

# Load models and tokenizers
@st.cache_resource
def load_models_and_tokenizers(model_id):
    tokenizer = AutoTokenizer.from_pretrained("../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_id=model_id)
    return model, tokenizer

# Define the Streamlit app
st.set_page_config(page_title="Code Selection Web Demo", page_icon=":robot_face:", layout="wide")
st.title("Code Selection Web Demo :robot_face:")

st.sidebar.header("模型配置")
option = st.sidebar.selectbox(
    '请选择一个要运行的模型:',
    ('命名实体识别专家', '法律支持', '法律文本续写','法律文书摘要生成')
)

st.sidebar.subheader("输入文本")
if option == '命名实体识别专家':
    input_value = st.sidebar.text_area("请输入需要进行命名实体识别的法律文本:")
elif option=='法律支持':
    input_value = st.sidebar.text_area("请输入您要咨询的法律问题：")
elif option == '法律文本续写':
    input_value = st.sidebar.text_area("请输入‘诉讼请求’和‘审理查明’部分内容，模型将回复生成‘本院认为’部分内容:")
elif option == '法律文书摘要生成':
    input_value = st.sidebar.text_area("请输入需要进行摘要的法律文书:")

if st.sidebar.button('运行'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with st.spinner('正在加载模型并生成答案'):
        if option == '命名实体识别专家':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/NER/checkpoint-1100")
            instruction = ("你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、"
                           "盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果："
                           "[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]")
        elif option=='法律支持':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/Lawer/checkpoint-400")
            instruction = ("假设你是一名律师，请回答下面这个真实情景下的中文法律咨询问题。")
        elif option == '法律文本续写':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/LDW/checkpoint-400")
            instruction = ("请你根据下面中括号里的'诉讼请求'和'审理查明'内容生成对应的'本院认为'内容。")
        elif option == '法律文书摘要生成':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/LTS/checkpoint-100")
            instruction = ("请对下面给的这篇法律文书提取摘要，用更短、更连贯、更自然的文字表达其主要内容。")
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer, device)
    st.success('回答生成成功!')
    st.subheader("模型回答如下")
    st.write(response)
else:
    st.info("请选择模型并输入内容，点击运行以获取回答。")

# Custom footer
st.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            color: white;
            text-align: center;
            padding: 10px;
            background: #004466;
        }
    </style>
    <div class="footer">
        <p>Developed by ZGZF&BUAA</p>
    </div>
    """, unsafe_allow_html=True)