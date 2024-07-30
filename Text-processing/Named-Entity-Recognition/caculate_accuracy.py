import json

def calculate_accuracy(prediction_file):
    # 加载预测结果
    with open(prediction_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    # 计算整体模型准确率
    correct_predictions = 0
    total_predictions = len(predictions)
    total_fields = 0
    correct_fields = 0

    for pred in predictions:
        pred_values = pred["predicted"].split("; ")
        ans_values = pred["expected"].split("; ")

        # 检查字段数是否一致
        assert len(pred_values) == len(ans_values), "预测和答案的字段数量不一致"

        # 检查整体预测是否完全正确
        if pred["predicted"] == pred["expected"]:
            correct_predictions += 1

        # 逐字段比较
        for pred_value, ans_value in zip(pred_values, ans_values):
            pred_label, pred_text = pred_value.split(": ")
            ans_label, ans_text = ans_value.split(": ")

            assert pred_label == ans_label, "预测和答案的标签不一致"

            total_fields += 1
            if pred_text == ans_text:
                correct_fields += 1

    # 计算整体模型准确率
    model_accuracy = correct_predictions / total_predictions

    # 计算字段匹配度百分比
    field_matching = correct_fields / total_fields

    return model_accuracy, field_matching


# 文件列表
prediction_files = ["./data/results_with_checkpoint.json", "./data/results_origin.json"]

# 处理每个文件并打印结果
for prediction_file in prediction_files:
    model_accuracy, field_matching = calculate_accuracy(prediction_file)
    print(f"Results for {prediction_file}:")
    print(f"  Model Accuracy: {model_accuracy * 100:.2f}%")
    print(f"  Field Matching Accuracy: {field_matching * 100:.2f}%\n")