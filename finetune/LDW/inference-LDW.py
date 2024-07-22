import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 加载原下载路径的tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("../../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("../../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16,trust_remote_code=True)

# 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
model = PeftModel.from_pretrained(model, model_id="../output/LDW/checkpoint-400")

test_texts = {
    'instruction': "请你根据下面中括号里的'诉讼请求'和'审理查明'内容生成对应的'本院认为'内容。",
    'input': """['诉讼请求'：请求依法判令被告陶浩军归还借款50000元，并支付自起诉之日起至实际履行之日止按中国人民银行同期贷款利率计算的逾期付款利息；判令被告殷建根对被告陶浩军的债务承担连带清偿责任；本案诉讼费由二被告承担。\n'审理查明'：经审理，本院认定如下事实：2014年1月4日被告陶浩军向原告陈森林借款人民币50000元，并由被告殷建根作担保人，当场出具借款合同一份，该借款由原告通过现金的形式直接支付给被告陶浩军。后该借款至今未归还，遂成讼。]"""
}

instruction = test_texts['instruction']
input_value = test_texts['input']

messages = [
    {"role": "system", "content": f"{instruction}"},
    {"role": "user", "content": f"{input_value}"}
]

response = predict(messages, model, tokenizer)
print(response)
