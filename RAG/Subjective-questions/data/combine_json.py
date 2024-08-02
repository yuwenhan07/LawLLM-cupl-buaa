import json

# Function to read and parse JSON files
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to format data into Markdown
def format_to_markdown(data, title):
    md_content = f"# {title}\n\n"
    
    if isinstance(data, dict):
        # If data is a dictionary, wrap it in a list
        data = [data]
    
    for entry in data:
        md_content += "## Entry\n\n"
        for key, value in entry.items():
            md_content += f"- **{key}**: {value}\n"
        md_content += "\n"
    
    return md_content

# Read data from JSON files
json_files = ['baseline.json', 'rag.json', 'mvrag.json']
titles = ['baseline', 'rag', 'mvrag']

# Aggregate Markdown content
markdown_content = ""
for file, title in zip(json_files, titles):
    data = read_json_file(file)
    markdown_content += format_to_markdown(data, title)
    markdown_content += "\n---\n"

# Write Markdown content to a file
with open('consolidated_data.md', 'w') as md_file:
    md_file.write(markdown_content)

print("Markdown file 'consolidated_data.md' created successfully.")