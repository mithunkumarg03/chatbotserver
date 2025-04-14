from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import os

app = Flask(__name__)

tokenizer = Tokenizer.from_file("tokenizer.json")
ort_session = ort.InferenceSession("model.onnx")
print("ONNX Input Names:", [i.name for i in ort_session.get_inputs()])


def generate_response(prompt):
    try:
        encoding = tokenizer.encode(prompt)
        input_ids = np.array([encoding.ids[:512]], dtype=np.int64)

        # Fix for missing attention_mask
        attention = encoding.attention_mask
        if attention is None:
            attention = [1] * len(encoding.ids)
        attention_mask = np.array([attention[:512]], dtype=np.int64)

        # Print for debug (optional)
        print("Input IDs:", input_ids)
        print("Attention Mask:", attention_mask)

        # Use actual model input names (update once you know)
        outputs = ort_session.run(
            None,
            {
                "input_ids": input_ids,  # update if your model uses a different name
                "attention_mask": attention_mask
            }
        )

        return tokenizer.decode(outputs[0][0].tolist(), skip_special_tokens=True)

    except Exception as e:
        print("Error during inference:", str(e))  # critical for logs
        return "Sorry, an error occurred while generating a response."


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


