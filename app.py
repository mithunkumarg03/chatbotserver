from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Add this in Render environment

def ask_huggingface_bot(message):
    API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    payload = {
        "inputs": {
            "text": message
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Sorry, I'm having trouble responding right now."

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    bot_response = ask_huggingface_bot(user_msg)
    return jsonify({"reply": bot_response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
