from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import os

app = Flask(__name__)

# Initialize ONNX runtime with low-memory config
ort_session = ort.InferenceSession(
    "model.onnx",
    providers=['CPUExecutionProvider'],
    sess_options=ort.SessionOptions()
)
ort_session.disable_fallback()  # Prevent memory spikes

tokenizer = Tokenizer.from_file("tokenizer.json")

def generate_response(prompt):
    # Dynamic truncation for long queries
    encoding = tokenizer.encode(prompt)
    input_ids = np.array([encoding.ids[:256]], dtype=np.int64)  # Strict truncation
    attention_mask = np.array([encoding.attention_mask[:256]], dtype=np.int64)
    
    # Low-memory inference
    outputs = ort_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    )
    return tokenizer.decode(outputs[0][0].tolist(), skip_special_tokens=True)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message', '')[:500]  # Length limit
        prompt = f"Medical Assistant: Provide concise, accurate health information.\n\nUser: {user_input}\nAssistant:"
        response = generate_response(prompt)
        return jsonify({"response": response[:1000]})  # Response length limit
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
