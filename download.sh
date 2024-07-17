#!/bin/bash

REPO_URL="https://www.wisemodel.cn/ZhipuAI/GLM-4-9B-Chat.git"
TARGET_DIR="./GLM-4-9B-Chat"
RETRY_DELAY=10 # 重试等待时间（秒）

while true; do
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Attempting to clone the repository..."
        git clone $REPO_URL $TARGET_DIR
        if [ $? -eq 0 ]; then
            echo "Repository cloned successfully."
            break
        else
            echo "Failed to clone the repository. Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
        fi
    else
        echo "Target directory already exists. Assuming repository is cloned."
        break
    fi
done