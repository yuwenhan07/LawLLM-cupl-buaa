import os
import logging
import warnings

# 指定使用的 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

# 抑制TensorFlow的日志
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 设置Transformers的日志等级为ERROR
logging.getLogger("transformers").setLevel(logging.ERROR)

# 抑制警告
warnings.filterwarnings("ignore")

# 设置PyTorch的日志等级为ERROR
logging.getLogger("torch").setLevel(logging.ERROR)



import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.huggingface import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import swanlab
import sys





# 文本预处理
def dataset_jsonl_transfer(origin_path, new_path):
    """
    将原始数据集转换为大模型微调所需数据格式的新数据集
    """
    # 设定一个messages储存每一条文本
    messages = []

    # 读取旧的JSONL文件【注：使用jsonl文件，对每一行进行处理】
    with open(origin_path, "r") as file:
        for line in file:
            # 解析每一行的json数据  具体处理方式根据内容所决定
            data = json.loads(line)
            context = data["context"]
            NER = data["entities"]
            # 此处设定instruction作为指令微调的指令
            message = {
                    "instruction": """你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果：[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]。""",
                    "input": f"法律文本: \"{context}\"",
                    "output": NER
            }
            messages.append(message)

    # 保存重构后的JSONL文件
    with open(new_path, "w", encoding="utf-8") as file:
        for message in messages:
            file.write(json.dumps(message, ensure_ascii=False) + "\n")

# 数据集预处理函数            
def process_func(example):
    """
    将数据集进行预处理
    """
    # 超参数，定义输入序列和响应序列的总长度
    MAX_LENGTH = 512
    # 初始化空列表，用于存储 input_ids（输入 ID）、attention_mask（注意力掩码）和 labels（标签）
    input_ids, attention_mask, labels = [], [], []
    
    # 构建指令和输入文本的序列
    instruction_text = (
        f"<|system|>\n你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果：[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]<|endoftext|>\n<|user|>\n{example['input']}<|endoftext|>\n<|assistant|>\n"
    )
    
    # 对instruction进行tokenizer  不添加特殊的分词符
    # tokenizer 是一个用于将文本转换为模型输入格式的工具。具体来说，它将输入文本转换为 token ids 和注意力掩码，以便模型可以理解和处理这些文本。
    instruction = tokenizer(instruction_text, add_special_tokens=False)
    
    # 对response进行tokenizer
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    # 构建input_ids，attention_mask和labels
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]

    # 构建用于模型输入的注意力掩码 (attention_mask) 序列
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 这行代码的作用是构建模型训练时的目标输出标签 (labels) 列表。具体来说，它通过拼接不同部分来创建一个与 input_ids 长度一致的标签序列，其中：
	# •	指令部分用 -100 填充，表示这些位置的损失将被忽略。 huggingface中表示计算损失忽略这一部分的内容
	# •	响应部分用实际的 token IDs 填充。
	# •	最后添加一个 tokenizer.pad_token_id，与 input_ids 对应。
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断处理
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 生成结果函数
def predict(messages, model, tokenizer):
    if torch.cuda.is_available():
        device=torch.device("cuda:{}".format(0))
    else:
        print("CUDA is unavailable")
        sys.exit(1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 使用聊天模式
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成响应
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    # 解码响应结果
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(response)
     
    return response



def split_jsonl_file(input_path, train_output_path, test_output_path, num_test_samples=10):
    """
    将输入的 JSONL 文件拆分为训练集和测试集，前面的数据作为训练集，最后 num_test_samples 条作为测试集。
    
    参数:
    - input_path (str): 输入 JSONL 文件路径
    - train_output_path (str): 输出训练集 JSONL 文件路径
    - test_output_path (str): 输出测试集 JSONL 文件路径
    - num_test_samples (int): 测试集的样本数量，默认值为 10
    """
    # 读取整个数据集
    with open(input_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # 拆分数据集：前面的数据作为训练集，最后 num_test_samples 条作为测试集
    train_lines = lines[:-num_test_samples]
    test_lines = lines[-num_test_samples:]

    # 保存训练集
    with open(train_output_path, "w", encoding="utf-8") as train_file:
        for line in train_lines:
            train_file.write(line)
    
    # 保存测试集
    with open(test_output_path, "w", encoding="utf-8") as test_file:
        for line in test_lines:
            test_file.write(line)
    
    print(f"训练集已保存到 {train_output_path}")
    print(f"测试集已保存到 {test_output_path}")




'''
•	model.enable_input_require_grads()：这是一个启用模型输入梯度的方法。在某些情况下，如使用梯度检查点（Gradient Checkpointing）时，需要启用输入梯度计算。
•	梯度检查点是一种节省显存的方法，尤其在处理大模型时。它通过在前向传播时存储一部分中间结果，减少显存使用，但在反向传播时需要重新计算这些中间结果。
'''
# 本地模型路径
local_model_path = "../GLM-4-9B-Chat"

# 加载本地模型权重并指定设备
device=torch.device("cuda:{}".format(0))
tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map={"": device}, torch_dtype=torch.bfloat16, trust_remote_code=True)
model.to(device)

# Transformers加载本地模型权重
# tokenizer = AutoTokenizer.from_pretrained(local_model_path, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

# 开启梯度检查点时，要执行该方法
model.enable_input_require_grads()

# 加载数据集
input_path = "./data/law/NER.jsonl"
train_dataset_path = "./data/law/NER_train_origin.jsonl"
test_dataset_path = "./data/law/NER_test_origin.jsonl"
split_jsonl_file(input_path, train_dataset_path, test_dataset_path)

train_jsonl_new_path = "./data/law/NER_train.jsonl"
test_jsonl_new_path = "./data/law/NER_test.jsonl"

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

# lora 配置
config = LoraConfig(
    # 任务类型，这里设置为 TaskType.CAUSAL_LM，表示这是一个因果语言模型任务。因果语言模型用于生成下一个单词（token），给定前面的单词序列。
    task_type=TaskType.CAUSAL_LM,
    # •	目标模块列表，指定了在模型中应用 LoRA 的模块名称。
	# •	"query_key_value": 可能是用于自注意力机制中的查询和键的矩阵。
	# •	"dense": 全连接层。
	# •	"dense_h_to_4h": 从隐藏层到 4 倍隐藏层的全连接层。
	# •	"activation_func": 激活函数。
	# •	"dense_4h_to_h": 从 4 倍隐藏层到隐藏层的全连接层。
	# •	这些模块通常是大型模型中计算密集型的部分，通过在这些模块中应用 LoRA，可以显著减少需要微调的参数数量。
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "activation_func", "dense_4h_to_h"],
    # 需要进行梯度训练
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩 秩越小，表示分解后的矩阵越小，参数量减少越多，但可能会损失一些模型的表示能力。
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理 适配矩阵的值会乘以 32。
    lora_dropout=0.1,  # Dropout 比例
)

model = get_peft_model(model, config)

args = TrainingArguments(
    # 指定模型训练输出的目录。训练过程中生成的检查点和其他输出文件将保存到这个目录。
    output_dir="./output/GLM4-NER-3",
    per_device_train_batch_size=8,
    # 梯度累积步骤数。模型在实际更新参数之前会累积 4 个批次的梯度，相当于有效批量大小为 8 * 4 = 32。这在显存有限的情况下尤为有用。
    gradient_accumulation_steps=4,
    # 日志记录
    logging_steps=10,
    # 训练轮次
    num_train_epochs=3,
    # 保存检查点数量
    save_steps=50,
    # 学习率。控制模型参数更新的步伐。设置为 0.0001，表示每次参数更新的步伐较小，这有助于模型稳定训练。
    learning_rate=1e-4,
    # 是否在每个节点上保存检查点。在分布式训练中，这一参数确保在每个节点上都保存检查点。
    save_on_each_node=True,
    # 是否启用梯度检查点。启用梯度检查点可以节省显存，但会增加计算开销。对于大模型，这通常是必要的，以避免显存不足。
    gradient_checkpointing=True,
    # 指定日志报告的目标。设置为 "none"，表示不报告日志到外部工具，如 TensorBoard、WandB 等。
    report_to="none",
)

swanlab_callback = SwanLabCallback(
    project="GLM4-NER-fintune",
    experiment_name="GLM4-9B-Chat",
    description="使用智谱GLM4-9B-Chat模型在法律命名实体识别数据集上微调。",
    config={
        "model": "GLM4-9B-Chat",
        "dataset": "法律命名实体识别",
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
