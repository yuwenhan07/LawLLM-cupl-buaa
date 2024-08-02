import json

# 读取输入 JSON 文件
with open('../data/evaluation-mvrag.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取 score 字段内容
scores = []

for entry in data:
    entry_id = entry['id']
    score = entry.get('score', {})
    
    score_entry = {'id': entry_id, 'score': score}
    scores.append(score_entry)

# 输出到新的 JSON 文件
with open('../data/mvrag-score.json', 'w', encoding='utf-8') as f:
    json.dump(scores, f, ensure_ascii=False, indent=4)