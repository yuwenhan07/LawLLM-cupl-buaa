import json

def json_to_jsonl(json_path, jsonl_path):
    """
    将JSON文件转换为JSONL文件
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    # 写入JSONL文件
    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

# 示例用法
json_path = 'val_data.json'
jsonl_path = 'NER_val.jsonl'
json_to_jsonl(json_path, jsonl_path)
