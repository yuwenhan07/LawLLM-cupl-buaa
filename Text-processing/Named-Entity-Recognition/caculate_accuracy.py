import json

def calculate_accuracy(prediction_file, answer_file):
    # 加载预测结果
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # 加载真实答案
    with open(answer_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)

    # 确保预测和答案数量一致
    assert len(predictions) == len(answers), "预测结果和真实答案的数量不一致"

    # 计算正确率
    correct_predictions = 0
    total_predictions = len(predictions)

    for pred, ans in zip(predictions, answers):
        if pred["predicted"] == ans["entities"]:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

# 示例用法
prediction_file = "./output/results_ft.json"
answer_file = "./data/data.json"
accuracy = calculate_accuracy(prediction_file, answer_file)
print(f"Model Accuracy: {accuracy * 100:.2f}%")