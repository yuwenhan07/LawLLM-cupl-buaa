import json

def dict_to_markdown_table_from_str(data_str):
    # Parse the string input into a Python list of dictionaries
    data = json.loads(data_str)
    
    # Creating the markdown table header
    markdown_table = "| Label | Text |\n"
    markdown_table += "|-------|------|\n"
    
    # Filling the markdown table rows
    for item in data:
        markdown_table += f"| {item['label']} | {item['text']} |\n"
    
    return markdown_table

# Sample string data provided by the user
data_str = '''[{'label': '犯罪嫌疑人', 'text': '侯某某'}, {'label': '物品价值', 'text': '人民币594.8元'}, {'label': '被盗物品', 'text': '黑色双禾牌电动车1辆'}, {'label': '时间', 'text': '2018年5月30日下午'}, {'label': '地点', 'text': '无锡市锡山区孝顺街道环城路158号对面车库'}]'''

# Replace single quotes with double quotes for valid JSON
data_str = data_str.replace("'", '"')

# Generate the markdown table
markdown_table = dict_to_markdown_table_from_str(data_str)
