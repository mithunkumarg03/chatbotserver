#!/bin/bash
# Health Chatbot Model Downloader (Robust Version)
# Save this EXACTLY as shown (don't copy-paste from browsers)

set -euo pipefail  # Strict error handling

MODEL_URL="https://huggingface.co/optimum/distilbert-base-uncased/resolve/main/model.onnx"
TOKENIZER_URL="https://huggingface.co/optimum/distilbert-base-uncased/resolve/main/tokenizer.json"

echo "⬇️ Downloading model files..."
if ! curl -L -o model.onnx "$MODEL_URL"; then
    echo "❌ Model download failed" >&2
    exit 1
fi

if ! curl -L -o tokenizer.json "$TOKENIZER_URL"; then
    echo "❌ Tokenizer download failed" >&2
    exit 1
fi

echo "🔍 Verifying files..."
file model.onnx | grep -q "ONNX" || { echo "❌ Invalid ONNX file" >&2; exit 1; }
file tokenizer.json | grep -q "JSON" || { echo "❌ Invalid JSON file" >&2; exit 1; }

echo -e "\n✅ Success! Downloaded:"
ls -lh model.onnx tokenizer.json
exit 0
