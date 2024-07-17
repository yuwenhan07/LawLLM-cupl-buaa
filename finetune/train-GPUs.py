import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import os
import swanlab

# 指定只使用 CUDA 设备 0 和 1
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"

def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    messages = []

    # 读取旧的JSONL文件
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据
            data = json.loads(line)
            context = data["context"]
            NER = data["entities"]
            message = {
                    "instruction": """你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果：[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]。""",
                    "input": f"文本: \"{context}\"",
                    "output": NER
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 384 
    input_ids, attention_mask, labels = [], [], []
    
    # 构建指令和输入文本的序列
    instruction_text = (
        """你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果：[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]"""
        f"{example['input']}\n\n"
    )
    
    # 对instruction进行tokenizer
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    
    # 对response进行tokenizer
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    # 构建input_ids，attention_mask和labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def predict(messages, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response

# 在modelscope上下载GLM模型到本地目录下
model_dir = snapshot_download("ZhipuAI/glm-4-9b-chat", cache_dir="./", revision="master")

# Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("./ZhipuAI/glm-4-9b-chat/", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./ZhipuAI/glm-4-9b-chat/", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 包装模型以使用多个GPU
model = torch.nn.DataParallel(model)

# 加载、处理数据集和测试集
train_dataset_path = "/home/yuwenhan/law-LLM/buaa/finetune/data/law/NER.jsonl"
test_dataset_path = "/home/yuwenhan/law-LLM/buaa/finetune/data/law/NER.jsonl"

train_jsonl_new_path = "/home/yuwenhan/law-LLM/buaa/finetune/data/law/NER_train_3.jsonl"
test_jsonl_new_path = "/home/yuwenhan/law-LLM/buaa/finetune/data/law/NER_test_3.jsonl"

if not os.path.exists(train_jsonl_new_path):
    dataset_jsonl_transfer(train_dataset_path, train_jsonl_new_path)
if not os.path.exists(test_jsonl_new_path):
    dataset_jsonl_transfer(test_dataset_path, test_jsonl_new_path)

# 检查数据集文件是否为空
if os.path.getsize(train_jsonl_new_path) == 0:
    raise ValueError(f"Training dataset {train_jsonl_new_path} is empty.")
if os.path.getsize(test_jsonl_new_path) == 0:
    raise ValueError(f"Testing dataset {test_jsonl_new_path} is empty.")

# 得到训练集
train_df = pd.read_json(train_jsonl_new_path, lines=True)
if train_df.empty:
    raise ValueError(f"Training DataFrame is empty.")
train_ds = Dataset.from_pandas(train_df)
train_dataset = train_ds.map(process_func, remove_columns=train_ds.column_names)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir="./output/NER-GPUs-test",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="GLM4-fintune",
    experiment_name="GLM4-9B-Chat",
    description="使用智谱GLM4-9B-Chat模型在zh_cls_fudan-news数据集上微调。",
    config={
        "model": "ZhipuAI/glm-4-9b-chat",
        "dataset": "huangjintao/zh_cls_fudan-news",
    },
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
)

trainer.train()

# 用测试集的前10条，测试模型
test_df = pd.read_json(test_jsonl_new_path, lines=True)[:10]

test_text_list = []
for index, row in test_df.iterrows():
    instruction = row['instruction']
    input_value = row['input']
    
    messages = [
        {"role": "system", "content": f"{instruction}"},
        {"role": "user", "content": f"{input_value}"}
    ]

    response = predict(messages, model, tokenizer)
    messages.append({"role": "assistant", "content": f"{response}"})
    result_text = f"{messages[0]}\n\n{messages[1]}\n\n{messages[2]}"
    test_text_list.append(swanlab.Text(result_text, caption=response))
    
swanlab.log({"Prediction": test_text_list})
swanlab.finish()
