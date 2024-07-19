import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    # 加载原下载路径的tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained("../../GLM-4-9B-Chat", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("../../GLM-4-9B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

    if load_checkpoint:
        # 加载训练好的Lora模型，将下面的checkpointXXX替换为实际的checkpoint文件名名称
        model = PeftModel.from_pretrained(model, model_id="../../finetune/output/NER/checkpoint-1100")

    test_data = [
            {"id": "a1", "context": "被害人报案总价值1500余元。", "entities": [{"label": "物品价值", "text": "1500余元"}]},
        {"id": "a2", "context": "2019年1月5日凌晨，被告人李某在东风小区砸碎被害人王某某的车窗玻璃，盗窃苹果手机一部。", "entities": [{"label": "犯罪嫌疑人", "text": "李某"}, {"label": "受害人", "text": "王某某"}, {"label": "被盗物品", "text": "苹果手机一部"}, {"label": "时间", "text": "2019年1月5日凌晨"}, {"label": "地点", "text": "东风小区"}]},
        {"id": "a3", "context": "被害人报案总价值2200余元。", "entities": [{"label": "物品价值", "text": "2200余元"}]},
        {"id": "a4", "context": "2018年5月12日夜间，被告人张某在华光小区盗窃电动车一辆。", "entities": [{"label": "犯罪嫌疑人", "text": "张某"}, {"label": "时间", "text": "2018年5月12日夜间"}, {"label": "地点", "text": "华光小区"}, {"label": "被盗物品", "text": "电动车一辆"}]},
        {"id": "a5", "context": "被害人报案总价值5000余元。", "entities": [{"label": "物品价值", "text": "5000余元"}]},
        {"id": "a6", "context": "2020年3月15日凌晨，被告人王某在富华小区砸碎被害人刘某某的车窗玻璃，盗窃现金500元。", "entities": [{"label": "犯罪嫌疑人", "text": "王某"}, {"label": "受害人", "text": "刘某某"}, {"label": "被盗物品", "text": "现金500元"}, {"label": "时间", "text": "2020年3月15日凌晨"}, {"label": "地点", "text": "富华小区"}]},
        {"id": "a7", "context": "被害人报案总价值1800余元。", "entities": [{"label": "物品价值", "text": "1800余元"}]},
        {"id": "a8", "context": "2021年4月10日凌晨，被告人李某在新村小区砸碎被害人周某某的车窗玻璃，盗窃笔记本电脑一台。", "entities": [{"label": "犯罪嫌疑人", "text": "李某"}, {"label": "受害人", "text": "周某某"}, {"label": "被盗物品", "text": "笔记本电脑一台"}, {"label": "时间", "text": "2021年4月10日凌晨"}, {"label": "地点", "text": "新村小区"}]},
        {"id": "a9", "context": "被害人报案总价值600元。", "entities": [{"label": "物品价值", "text": "600元"}]},
        {"id": "a10", "context": "2017年9月25日夜间，被告人王某在康乐小区盗窃电动自行车一辆。", "entities": [{"label": "犯罪嫌疑人", "text": "王某"}, {"label": "时间", "text": "2017年9月25日夜间"}, {"label": "地点", "text": "康乐小区"}, {"label": "被盗物品", "text": "一辆电动自行车"}]},
        {"id": "a11", "context": "被害人报案总价值3200余元。", "entities": [{"label": "物品价值", "text": "3200余元"}]},
        {"id": "a12", "context": "2022年6月8日凌晨，被告人张某某在翠园小区砸碎被害人陈某某的车窗玻璃，盗窃现金300元。", "entities": [{"label": "犯罪嫌疑人", "text": "张某某"}, {"label": "受害人", "text": "陈某某"}, {"label": "被盗物品", "text": "现金300元"}, {"label": "时间", "text": "2022年6月8日凌晨"}, {"label": "地点", "text": "翠园小区"}]},
        {"id": "a13", "context": "被害人报案总价值4500余元。", "entities": [{"label": "物品价值", "text": "4500余元"}]},
        {"id": "a14", "context": "2019年12月31日夜间，被告人李某某在南湖小区盗窃一辆摩托车。", "entities": [{"label": "犯罪嫌疑人", "text": "李某某"}, {"label": "时间", "text": "2019年12月31日夜间"}, {"label": "地点", "text": "南湖小区"}, {"label": "被盗物品", "text": "摩托车一辆"}]},
        {"id": "a15", "context": "被害人报案总价值1200余元。", "entities": [{"label": "物品价值", "text": "1200余元"}]},
        {"id": "a16", "context": "2021年11月19日凌晨，被告人杨某在长兴小区砸碎被害人孙某某的车窗玻璃，盗窃金项链一条。", "entities": [{"label": "犯罪嫌疑人", "text": "杨某"}, {"label": "受害人", "text": "孙某某"}, {"label": "被盗物品", "text": "金项链一条"}, {"label": "时间", "text": "2021年11月19日凌晨"}, {"label": "地点", "text": "长兴小区"}]},
        {"id": "a17", "context": "被害人报案总价值750元。", "entities": [{"label": "物品价值", "text": "750元"}]},
        {"id": "a18", "context": "2018年8月14日夜间，被告人刘某某在福安小区盗窃一辆自行车。", "entities": [{"label": "犯罪嫌疑人", "text": "刘某某"}, {"label": "时间", "text": "2018年8月14日夜间"}, {"label": "地点", "text": "福安小区"}, {"label": "被盗物品", "text": "自行车一辆"}]},
        {"id": "a19", "context": "被害人报案总价值2600余元。", "entities": [{"label": "物品价值", "text": "2600余元"}]},
        {"id": "a20", "context": "2017年11月5日凌晨，被告人王某某在蓝天小区砸碎被害人赵某某的车窗玻璃，盗窃平板电脑一台。", "entities": [{"label": "犯罪嫌疑人", "text": "王某某"}, {"label": "受害人", "text": "赵某某"}, {"label": "被盗物品", "text": "平板电脑一台"}, {"label": "时间", "text": "2017年11月5日凌晨"}, {"label": "地点", "text": "蓝天小区"}]}
    ]

    correct_predictions = 0
    total_predictions = 0

    for test_instance in test_data:
        instruction = "你是一个法律命名实体识别的专家。请根据给定文本，从以下十个方面（犯罪嫌疑人、受害人、被盗货币、物品价值、盗窃获利、被盗物品、作案工具、时间、地点、组织机构）提取文中的实体，没有用None表示，并按照以下格式返回结果：[犯罪嫌疑人: xxx; 受害人： xxx; 被盗货币： None; ……]"
        input_value = test_instance["context"]

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{input_value}"}
        ]

        response = predict(messages, model, tokenizer)
        
        # 将实际标签格式化为字符串
        expected_response = []
        for entity in ["犯罪嫌疑人", "受害人", "被盗货币", "物品价值", "盗窃获利", "被盗物品", "作案工具", "时间", "地点", "组织机构"]:
            entity_values = [e["text"] for e in test_instance["entities"] if e["label"] == entity]
            expected_response.append(f"{entity}: {entity_values[0] if entity_values else 'None'}")
        expected_response = "; ".join(expected_response)
        
        # 解析和格式化预测结果
        parsed_response = parse_response(response)
        formatted_response = format_response(parsed_response)

        print(f"Predicted: {formatted_response}")
        print(f"Expected: {expected_response}")

        # 进行简单的字符串比较来验证预测结果
        if formatted_response == expected_response:
            correct_predictions += 1
        total_predictions += 1

    accuracy = correct_predictions / total_predictions
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_checkpoint", action="store_true", help="Whether to load a trained LoRA checkpoint")
    args = parser.parse_args()

    main(args.load_checkpoint)