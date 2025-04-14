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
        attention_mask = np.array([encoding.attention_mask[:512]], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        outputs = ort_session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )

        # Assuming output is logits (shape: [1, seq_len, vocab_size])
        output_logits = outputs[0]
        token_ids = np.argmax(output_logits, axis=-1)[0]  # Get the predicted token ids

        response_text = tokenizer.decode(token_ids.tolist())
        return response_text

    except Exception as e:
        import traceback
        traceback.print_exc()
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


