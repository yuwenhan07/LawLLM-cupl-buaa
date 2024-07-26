import requests
from tqdm import tqdm
import os

# 确保目标文件夹存在
os.makedirs('BAAI_bge-m3', exist_ok=True)

# Base URL of the Hugging Face repository
base_url = "https://huggingface.co/BAAI/bge-m3/resolve/main/"

# List of files to download
files_to_download = [
    "config.json",
    "config_sentence_transformers.json",
    "long.jpg",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "colbert_linear.pt",
    "sparse_linear.pt"
]

# Function to download a file with a progress bar
def download_file(filename):
    url = base_url + filename
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        with open(os.path.join('downloads', filename), 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename}")

# Download each file
for file in files_to_download:
    download_file(file)