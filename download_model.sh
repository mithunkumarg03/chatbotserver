#!/bin/bash
# Download 22MB optimized ONNX model and tokenizer
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.onnx -O model.onnx
wget https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json -O tokenizer.json

# Verify downloads
ls -lh model.onnx tokenizer.json