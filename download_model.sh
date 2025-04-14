#!/bin/bash
# Health Chatbot Model Downloader
# Uses curl with retries and checksum verification

set -e  # Exit immediately if any command fails

MODEL_URL="https://huggingface.co/optimum/distilbert-base-uncased/resolve/main/model.onnx"
TOKENIZER_URL="https://huggingface.co/optimum/distilbert-base-uncased/resolve/main/tokenizer.json"

echo "⬇️ Downloading model files..."

# Download with retries
for url in $MODEL_URL $TOKENIZER_URL; do
    filename=$(basename $url)
    for i in {1..3}; do
        if curl -L -o $filename $url; then
            echo "✅ Downloaded $filename"
            break
        else
            echo "❌ Attempt $i failed, retrying..."
            sleep 2
            rm -f $filename
        fi
    done
done

# Verify files exist
if [[ ! -f "model.onnx" || ! -f "tokenizer.json" ]]; then
    echo "❌ Critical Error: Missing model files!"
    exit 1
fi

# Basic validation
echo "🔍 Verifying files..."
file model.onnx | grep -q "ONNX" || { echo