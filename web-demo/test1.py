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
@st.cache(allow_output_mutation=True)
def load_models_and_tokenizers(model_id):
    tokenizer = AutoTokenizer.from_pretrained("../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, model_id=model_id)
    return model, tokenizer

# Define the Streamlit app
st.title("Code Selection Web Demo")
option = st.selectbox(
    'Select the code to run:',
    ('Code 1: NER Expert', 'Code 2: Legal Consultant')
)

input_value = st.text_area("Enter the text content:")

if st.button('Run Code'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if option == 'Code 1: NER Expert':
        model, tokenizer = load_models_and_tokenizers("../finetune/output/NER/checkpoint-1100")
        instruction = ("你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、"
                       "盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果："
                       "[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]")
    else:
        model, tokenizer = load_models_and_tokenizers("../finetune/output/Lawer/checkpoint-400")
        instruction = ("假设你是一名律师，请回答下面这个真实情景下的中文法律咨询问题。")

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer, device)
    st.write(response)