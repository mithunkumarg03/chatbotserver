from transformers import PreTrainedTokenizerFast
import onnxruntime as ort
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Provide the directory path containing tokenizer files (including tokenizer.json)
tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer.json")

# Initialize ONNX session
ort_session = ort.InferenceSession("model.onnx")
print("ONNX Input Names:", [i.name for i in ort_session.get_inputs()])

def generate_response(prompt):
    try:
        encoding = tokenizer(prompt, return_tensors="np", padding="max_length", max_length=512, truncation=True)
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        token_type_ids = encoding.get("token_type_ids", np.zeros_like(input_ids))

        # Run inference with ONNX model
        outputs = ort_session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            }
        )

        # Assuming model returns logits of shape [1, seq_len, vocab_size]
        output_logits = outputs[0]
        token_ids = np.argmax(output_logits, axis=-1)[0]

        # Decode the generated tokens
        response_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        return response_text

    except Exception as e:
        import traceback
        traceback.print_exc()
        return "Sorry, an error occurred while generating a response."

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', '')
    prompt = f"""As a medical AI assistant, provide concise, evidence-based responses.\n\nUser: {user_input}\nAssistant:"""

    response = generate_response(prompt)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
