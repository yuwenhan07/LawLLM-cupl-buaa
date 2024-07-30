import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse



def predict(messages, model, tokenizer):
    device = "cuda"

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

def parse_response(response):
    try:
        response_data = eval(response)
        if isinstance(response_data, list):
            return response_data
        else:
            return [response_data]
    except:
        return []

def format_response(parsed_response):
    formatted_response = []
    for entity in ["犯罪嫌疑人", "受害人", "被盗货币", "物品价值", "盗窃获利", "被盗物品", "作案工具", "时间", "地点", "组织机构"]:
        entity_values = [e["text"] for e in parsed_response if e["label"] == entity]
        formatted_response.append(f"{entity}: {entity_values[0] if entity_values else 'None'}")
    return "; ".join(formatted_response)

def main(load_checkpoint):
    # 加载原下载路径的tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained("../../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained("../../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

    if load_checkpoint:
        # 加载 fine-tuned 模型
        finetune_model = PeftModel.from_pretrained(base_model, model_id="../../finetune/output/NER/checkpoint-1100")
    else:
        finetune_model = None

    # 从指定路径加载测试数据
    data_path = "./data/data.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 预测并保存 baseline 模型的结果
    base_results = []
    for test_instance in test_data:
        instruction = "你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果：[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]"
        input_value = test_instance["context"]

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, base_model, tokenizer)
        parsed_response = parse_response(response)
        formatted_response = format_response(parsed_response)

        # 将实际标签格式化为字符串
        expected_response = []
        for entity in ["犯罪嫌疑人", "受害人", "被盗货币", "物品价值", "盗窃获利", "被盗物品", "作案工具", "时间", "地点", "组织机构"]:
            entity_values = [e["text"] for e in test_instance["entities"] if e["label"] == entity]
            expected_response.append(f"{entity}: {entity_values[0] if entity_values else 'None'}")
        expected_response = "; ".join(expected_response)

        base_results.append({
            "input": input_value,
            "predicted": formatted_response,
            "expected": expected_response
        })

    # 保存 baseline 模型的结果
    os.makedirs("./output", exist_ok=True)
    with open("./output/results_baseline.json", "w", encoding="utf-8") as f:
        json.dump(base_results, f, ensure_ascii=False, indent=4)

    # 计算 baseline 模型的正确率
    base_correct_predictions = sum(1 for result in base_results if result["predicted"] == result["expected"])
    base_total_predictions = len(base_results)
    base_accuracy = base_correct_predictions / base_total_predictions
    print(f"Baseline Model Accuracy: {base_accuracy * 100:.2f}%")

    # 如果加载了 fine-tuned 模型，则预测并保存其结果
    if finetune_model:
        ft_results = []
        for test_instance in test_data:
            response = predict(messages, finetune_model, tokenizer)
            parsed_response = parse_response(response)
            formatted_response = format_response(parsed_response)

            expected_response = []
            for entity in ["犯罪嫌疑人", "受害人", "被盗货币", "物品价值", "盗窃获利", "被盗物品", "作案工具", "时间", "地点", "组织机构"]:
                entity_values = [e["text"] for e in test_instance["entities"] if e["label"] == entity]
                expected_response.append(f"{entity}: {entity_values[0] if entity_values else 'None'}")
            expected_response = "; ".join(expected_response)

            ft_results.append({
                "input": input_value,
                "predicted": formatted_response,
                "expected": expected_response
            })

        # 保存 fine-tuned 模型的结果
        with open("./output/results_finetune.json", "w", encoding="utf-8") as f:
            json.dump(ft_results, f, ensure_ascii=False, indent=4)

        # 计算 fine-tuned 模型的正确率
        ft_correct_predictions = sum(1 for result in ft_results if result["predicted"] == result["expected"])
        ft_total_predictions = len(ft_results)
        ft_accuracy = ft_correct_predictions / ft_total_predictions
        print(f"Fine-Tuned Model Accuracy: {ft_accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", action="store_true", help="Whether to load a trained LoRA checkpoint")
    args = parser.parse_args()

    main(args.load_checkpoint)