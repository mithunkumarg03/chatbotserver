from transformers import PreTrainedTokenizerFast
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Initialize tokenizer (ensure tokenizer.json is in the current directory or give full path)
tokenizer_path = os.path.join(os.getcwd(), "tokenizer.json")
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

# Initialize ONNX session
onnx_model_path = os.path.join(os.getcwd(), "model.onnx")
ort_session = ort.InferenceSession(onnx_model_path)

# Print ONNX input names
onnx_inputs = {i.name for i in ort_session.get_inputs()}
print("ONNX Input Names:", onnx_inputs)

def generate_response(prompt):
    try:
        encoding = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=512, truncation=True)
        input_feed = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"]
        }

        # Add token_type_ids if required by the model
        if "token_type_ids" in onnx_inputs:
            input_feed["token_type_ids"] = encoding.get("token_type_ids", np.zeros_like(encoding["input_ids"]))

        # Run inference with ONNX model
        outputs = ort_session.run(None, input_feed)

        # Assuming outputs[0] = logits -> shape: [1, seq_len, vocab_size]
        output_logits = outputs[0]
        token_ids = np.argmax(output_logits, axis=-1)[0]

        # Decode tokens
        response_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return response_text.strip()

    except Exception as e:
        import traceback
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
