import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re
from random import shuffle

os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# 定义预测函数
def predict(messages, model, tokenizer, device):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

# 输出去重函数
def remove_duplicates(data):
    seen = set()
    unique_data = []
    for item in data:
        item_tuple = (item['label'], item['text'])
        if item_tuple not in seen:
            seen.add(item_tuple)
            unique_data.append(item)
    return unique_data

# 加载模型和tokenizer
@st.cache_resource
def load_models_and_tokenizers(model_id):
    tokenizer = AutoTokenizer.from_pretrained("../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_id=model_id)
    return model, tokenizer

# 数据预处理函数
def data_process(text):
    # 定义标点符号和特殊字母
    punctuation = '''，。、:；（）ＸX×xa"“”,<《》'''
    line1 = re.sub(u"（.*?）", "", text)  # 去除括号内注释
    line2 = re.sub("[%s]+" % punctuation, "", line1)  # 去除标点、特殊字母
    return line2

# 定义Streamlit应用
st.set_page_config(page_title="Law-llm", page_icon=":robot_face:", layout="wide")
st.title("Lawllm :robot_face:")

# 添加自定义CSS样式
st.markdown("""
    <style>
    /* 设置主要内容区的背景颜色、内边距和边框圆角 */
    .main {
        background-color: #FFFFFF;  /* 设置背景颜色为白色 */
        padding: 20px;  /* 内边距为20像素 */
        border-radius: 10px;  /* 边框圆角半径为10像素 */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* 添加轻微阴影 */
    }
    
    /* 设置侧边栏内容区的背景颜色和边框圆角 */
    .sidebar .sidebar-content {
        background-color: #dfe6f0;  /* 设置背景颜色为浅灰蓝色 */
        border-radius: 10px;  /* 边框圆角半径为10像素 */
        padding: 10px; /* 添加内边距 */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); /* 添加轻微阴影 */
    }
    
    /* 设置页脚的样式，使其固定在页面底部 */
    .footer {
        position: fixed;  /* 固定定位 */
        bottom: 0;  /* 贴近页面底部 */
        width: 100%;  /* 宽度占满页面 */
        color: white;  /* 文字颜色为白色 */
        text-align: center;  /* 文字居中对齐 */
        padding: 10px;  /* 内边距为10像素 */
        background: #2583E5;  /* 背景颜色为深蓝色 */
        box-shadow: 0px -2px 6px rgba(0, 0, 0, 0.1); /* 添加上方阴影 */
    }
    
    /* 设置按钮的样式 */
    .stButton>button {
        background-color: #2583E5;  /* 按钮背景颜色为深蓝色 */
        color: white;  /* 文字颜色为白色 */
        border-radius: 10px;  /* 边框圆角半径为10像素 */
        padding: 10px 20px;  /* 上下内边距为10像素，左右内边距为20像素 */
        font-size: 16px;  /* 字体大小为16像素 */
        margin: 10px 0;  /* 上下外边距为10像素 */
        border: none;  /* 去除默认边框 */
        cursor: pointer; /* 鼠标悬停时显示为手指 */
        transition: background-color 0.3s ease; /* 添加平滑的背景色变化效果 */
    }

    .stButton>button:hover {
        background-color: #1e6bb8;  /* 鼠标悬停时背景颜色变为更深的蓝色 */
    }
    
    /* 设置文本区域的样式 */
    .stTextArea textarea {
        border-radius: 10px;  /* 边框圆角半径为10像素 */
        border: 1px solid #ccc; /* 设置边框颜色为浅灰色 */
        padding: 10px; /* 添加内边距 */
        box-shadow: inset 0px 1px 3px rgba(0, 0, 0, 0.1); /* 添加内部阴影 */
    }
    
    /* 设置输出结果的字体大小 */
    .large-font {
        font-size: 20px;  /* 设置字体大小为20像素 */
        line-height: 1.5; /* 设置行高 */
        color: #333;  /* 设置文字颜色为深灰色 */
    }
</style>
""", unsafe_allow_html=True)


# 初始化会话状态
if 'processed_cases' not in st.session_state:
    st.session_state['processed_cases'] = ''
if 'copy_processed' not in st.session_state:
    st.session_state['copy_processed'] = False
if 'preprocess_done' not in st.session_state:
    st.session_state['preprocess_done'] = False

# 数据预处理UI
st.sidebar.header("数据预处理")
input_text = st.sidebar.text_area("请输入需要进行预处理的文本（去除标点、注释等信息）:")

if st.sidebar.button('进行数据预处理'):
    with st.spinner('正在进行数据预处理...'):
        st.session_state['processed_cases'] = data_process(input_text)
    st.success('数据预处理完成！')
    st.subheader("预处理后的数据")
    st.write(st.session_state['processed_cases'])
    st.session_state['preprocess_done'] = True
    st.session_state['copy_processed'] = False

if st.session_state['preprocess_done']:
    if st.button('一键复制'):
        st.session_state['copy_processed'] = True
        st.session_state['preprocess_done'] = False
        st.rerun()

st.sidebar.subheader("模型配置")
option = st.sidebar.selectbox(
    '请选择一个要运行的模型:',
    ('命名实体识别专家', '法律支持', '法律文本续写','法律文书摘要生成')
)

st.sidebar.subheader("输入文本")
input_area_default = st.session_state['processed_cases'] if st.session_state['copy_processed'] else ""
if option == '命名实体识别专家':
    input_value = st.sidebar.text_area("请输入需要进行命名实体识别的法律文本:", value=input_area_default)
elif option == '法律支持':
    input_value = st.sidebar.text_area("请输入您要咨询的法律问题：", value=input_area_default)
elif option == '法律文本续写':
    input_value = st.sidebar.text_area("请输入‘诉讼请求’和‘审理查明’部分内容，模型将回复生成‘本院认为’部分内容:", value=input_area_default)
elif option == '法律文书摘要生成':
    input_value = st.sidebar.text_area("请输入需要进行摘要的法律文书:", value=input_area_default)

if st.sidebar.button('运行'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with st.spinner('正在加载模型并生成答案'):
        if option == '命名实体识别专家':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/NER/checkpoint-800")
            instruction = ("你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、"
                           "盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果："
                           "[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]")
        elif option == '法律支持':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/Lawer/checkpoint-400")
            instruction = ("假设你是一名律师，请回答下面这个真实情景下的中文法律咨询问题。")
        elif option == '法律文本续写':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/LDW/checkpoint-300")
            instruction = ("请你根据下面中括号里的'诉讼请求'和'审理查明'内容生成对应的'本院认为'内容。")
        elif option == '法律文书摘要生成':
            model, tokenizer = load_models_and_tokenizers("../finetune/output/LTS/checkpoint-100")
            instruction = ("请对下面给的这篇法律文书提取摘要，用更短、更连贯、更自然的文字表达其主要内容。")
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer, device)
    if option == '命名实体识别专家':
        st.success('回答生成成功!')
        st.subheader("法律文本：")
        st.markdown(f'<div class="large-font">{input_value}</div>', unsafe_allow_html=True)
        st.subheader("命名实体结果：")
        st.markdown(f'<div class="large-font">{response}</div>', unsafe_allow_html=True)
    elif option == '法律支持':
        st.success('回答生成成功!')
        st.subheader("法律问题：")
        st.markdown(f'<div class="large-font">{input_value}</div>', unsafe_allow_html=True)
        st.subheader("法律支持回答：")
        st.markdown(f'<div class="large-font">{response}</div>', unsafe_allow_html=True)
    elif option == '法律文本续写':
        st.success('回答生成成功!')
        st.subheader("诉讼请求和审理查明：")
        st.markdown(f'<div class="large-font">{input_value}</div>', unsafe_allow_html=True)
        st.subheader("本院认为：")
        response = response + "。"
        st.markdown(f'<div class="large-font">{response}</div>', unsafe_allow_html=True)
    elif option == '法律文书摘要生成':
        st.success('回答生成成功!')
        st.subheader("法律长文本：")
        st.markdown(f'<div class="large-font">{input_value}</div>', unsafe_allow_html=True)
        st.subheader("文本摘要：")
        st.markdown(f'<div class="large-font">{response}</div>', unsafe_allow_html=True)
else:
    st.info("请选择模型功能并输入内容，点击运行以获取回答。")

# Custom footer
footer_html = """
    <div class="footer">
        <p>Developed by ZGZF&BUAA</p>
    </div>
"""
st.markdown(footer_html, unsafe_allow_html=True)