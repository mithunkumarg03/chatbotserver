from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify
import os
import traceback

app = Flask(__name__)

# Load tokenizer directly from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # 🔁 Change if you used another model

# Load ONNX model
onnx_model_path = os.path.join(os.getcwd(), "model.onnx")
ort_session = ort.InferenceSession(onnx_model_path)

# Get ONNX input names
onnx_inputs = {i.name for i in ort_session.get_inputs()}
print("ONNX Input Names:", onnx_inputs)

def generate_response(prompt):
    try:
        encoding = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=512, truncation=True)

        input_feed = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"]
        }

        if "token_type_ids" in onnx_inputs:
            input_feed["token_type_ids"] = encoding.get("token_type_ids", np.zeros_like(encoding["input_ids"]))

        outputs = ort_session.run(None, input_feed)

        logits = outputs[0]  # shape: [1, seq_len, vocab_size]
        token_ids = np.argmax(logits, axis=-1)[0]

        response = tokenizer.decode(token_ids, skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        traceback.print_exc()
        return "Sorry, an error occurred while generating a response."

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    prompt = f"""As a medical AI assistant, provide concise, evidence-based responses.\n\nUser: {user_input}\nAssistant:"""
    response = generate_response(prompt)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server Running"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
