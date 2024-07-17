import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from flask import Flask, request, jsonify, render_template

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

app = Flask(__name__)

# Load the tokenizer and model from the specified path
tokenizer = AutoTokenizer.from_pretrained(
    "/home/yuwenhan/law-LLM/buaa&zgzf/finetune/ZhipuAI/glm-4-9b-chat/",
    use_fast=False,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/home/yuwenhan/law-LLM/buaa&zgzf/finetune/ZhipuAI/glm-4-9b-chat/",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Load the fine-tuned Lora model
model = PeftModel.from_pretrained(model, model_id="/home/yuwenhan/law-LLM/buaa&zgzf/finetune/output/GLM4-NER-2/checkpoint-200")
model.to("cuda")

def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def response_to_table(response):
    # Convert response to a list of dictionaries
    data = [item.split(": ") for item in response.strip('[]').split("; ")]
    data = [{"label": item[0].strip(), "text": item[1].strip()} for item in data]

    # Convert the list of dictionaries to an HTML table
    table = "<table border='1'><tr><th>Label</th><th>Text</th></tr>"
    for item in data:
        table += f"<tr><td>{item['label']}</td><td>{item['text']}</td></tr>"
    table += "</table>"
    
    return table

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    data = request.json
    input_value = data['input']
    instruction = "你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果：[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]"

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    response_table = response_to_table(response)

    return response_table

if __name__ == '__main__':
    app.run(debug=True)
