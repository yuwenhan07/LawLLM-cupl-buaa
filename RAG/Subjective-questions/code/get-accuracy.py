import json

def extract_relevant_data(json_file_path, output_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
        # 确保数据是列表
        if not isinstance(data, list):
            raise ValueError("JSON data is not a list")
        
        extracted_data = []
        for item in data:
            extracted_item = {
                'id': item.get('id'),
                'total': item.get('score', {}).get('total'),
                'get': item.get('score', {}).get('get'),
                'ratio': item.get('ratio')
            }
            extracted_data.append(extracted_item)
        
        with open(output_file_path, 'w') as outfile:
            json.dump(extracted_data, outfile, indent=4)

if __name__ == "__main__":
    input_json_file_path = '../data/mvrag-ratios.json'  # 替换为你的输入 JSON 文件路径
    output_json_file_path = '../data/mvrag.json'  # 替换为你的输出 JSON 文件路径
    
    extract_relevant_data(input_json_file_path, output_json_file_path)
    
    print(f"Extracted data has been saved to {output_json_file_path}")