from flask import Flask, request, jsonify
from transformers import pipeline, Conversation

# Load Hugging Face chatbot pipeline
chatbot = pipeline("conversational", model="microsoft/DialoGPT-small")

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    try:
        conv = Conversation(user_message)
        result = chatbot(conv)
        reply = result.generated_responses[-1]
        return jsonify({"response": reply})
    except Exception:
        return jsonify({"error": "Server error"}), 500

@app.route("/", methods=["GET"])
def home():
    return "Chatbot Transformer Server Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
