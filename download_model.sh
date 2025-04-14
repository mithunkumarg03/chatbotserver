#!/bin/bash
set -euo pipefail

# Updated reliable sources (October 2023 verified)
MODEL_URL="https://github.com/onnx/models/raw/main/text/machine_comprehension/distilbert/model/distilbert-base-uncased.onnx"
TOKENIZER_URL="https://media.githubusercontent.com/media/onnx/models/main/text/machine_comprehension/distilbert/model/tokenizer.json"

# Enhanced download function
download_file() {
    local url="$1" filename="$2"
    echo "Downloading $filename..."
    
    for attempt in {1..5}; do
        # Use -f to fail silently on server errors
        if curl -fL -o "$filename" "$url" && [ "$(file -b --mime-type "$filename")" = "application/octet-stream" ]; then
            return 0
        fi
        
        echo "Attempt $attempt failed - $(file -b "$filename")" >&2
        rm -f "$filename"
        sleep $((attempt * 2))  # Exponential backoff
    done
    
    echo "❌ Failed after 5 attempts - last URL: $url" >&2
    exit 1
}

# Main execution
download_file "$MODEL_URL" "model.onnx"
download_file "$TOKENIZER_URL" "tokenizer.json"

# Final verification
echo "✅ Valid files downloaded:"
file model.onnx tokenizer.json
ls -lh model.onnx tokenizer.json
