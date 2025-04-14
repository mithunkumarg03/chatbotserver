from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer

app = Flask(__name__)

tokenizer = Tokenizer.from_file("tokenizer.json")
ort_session = ort.InferenceSession("model.onnx")

def generate_response(prompt):
    encoding = tokenizer.encode(prompt)
    input_ids = np.array([encoding.ids[:512]], dtype=np.int64)
    attention_mask = np.array([encoding.attention_mask[:512]], dtype=np.int64)

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
    user_input = request.json.get('message', '')
    prompt = f"""As a medical AI assistant, provide concise, evidence-based responses.

User: {user_input}
Assistant:"""

    response = generate_response(prompt)
    return jsonify({"response": response})


@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server Running"
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


