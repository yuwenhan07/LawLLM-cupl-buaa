import ast

# 你的字符串
data_str = "[{'label': '犯罪嫌疑人', 'text': '李小明'}, {'label': '犯罪嫌疑人', 'text': '李小明'}, {'label': '犯罪嫌疑人', 'text': '李小明'}, {'label': '犯罪嫌疑人', 'text': '李小明'}, {'label': '受害人', 'text': '张强'}, {'label': '受害人', 'text': '张强'}, {'label': '时间', 'text': '2024年6月15日'}, {'label': '地点', 'text': '北京市朝阳区三里屯'}, {'label': '组织机构', 'text': '北京市朝阳区人民法院'}, {'label': '组织机构', 'text': '北京市人民检察院朝阳区分院'}, {'label': '组织机构', 'text': '北京市华龙律师事务所'}]"

# 使用 ast.literal_eval 将字符串转换为列表
data_list = ast.literal_eval(data_str)


def remove_duplicates(data):
    seen = set()
    unique_data = []
    for item in data:
        item_tuple = (item['label'], item['text'])
        if item_tuple not in seen:
            seen.add(item_tuple)
            unique_data.append(item)
    return unique_data

unique_data = remove_duplicates(data_list)
print(unique_data)


# 打印转换后的列表
# print(data_list)
