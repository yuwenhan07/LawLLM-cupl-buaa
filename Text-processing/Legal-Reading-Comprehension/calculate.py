import json
from difflib import SequenceMatcher

def calculate_similarity(s1, s2):
    """计算两个字符串之间的相似度"""
    return SequenceMatcher(None, s1, s2).ratio()

def compare_responses_and_answers(response_file, answer_file):
    # 加载预测结果和期望答案
    responses = []
    with open(response_file, 'r', encoding='utf-8') as f:
        for line in f:
            responses.append(json.loads(line))
            
    answers = []
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            answers.append(json.loads(line))

    # 检查长度是否一致
    assert len(responses) == len(answers), "两个文件的记录数不一致"

    total_records = len(responses)
    correct_predictions = 0
    total_similarity = 0
    similarity_count = 0

    for response, answer in zip(responses, answers):
        predicted_response = response.get("response", "").strip()
        expected_answer = answer.get("answer", "").strip()

        # 检查整体预测是否完全正确
        if predicted_response == expected_answer:
            correct_predictions += 1

        # 计算相似度
        if predicted_response and expected_answer:
            similarity = calculate_similarity(predicted_response, expected_answer)
            total_similarity += similarity
            similarity_count += 1

    # 计算整体模型准确率
    model_accuracy = correct_predictions / total_records

    # 计算平均相似度
    avg_similarity = total_similarity / similarity_count if similarity_count > 0 else 0

    return model_accuracy, avg_similarity

# 文件路径
response_files = {
    "Model with Checkpoint": "./data/model_responses.jsonl",
    "Baseline Model": "./data/baseline_responses.jsonl"
}
answer_file = "./data/test.jsonl"

# 对每个文件进行比较并打印结果
for name, response_file in response_files.items():
    model_accuracy, avg_similarity = compare_responses_and_answers(response_file, answer_file)
    print(f"Results for {name}:")
    print(f"  Model Accuracy: {model_accuracy * 100:.2f}%")
    print(f"  Average Similarity: {avg_similarity * 100:.2f}%\n")