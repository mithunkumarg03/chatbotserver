import os
import traceback
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Add this in Render environment

def ask_huggingface_bot(message):
    try:
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
    except Exception as e:
        # Log the error traceback
        error_message = f"An error occurred while trying to fetch the response: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return "Sorry, I'm having trouble responding right now."

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_msg = request.json.get("message", "")
        bot_response = ask_huggingface_bot(user_msg)
        return jsonify({"reply": bot_response})
    except Exception as e:
        # Log the error traceback for any issues in the /chat route
        error_message = f"An error occurred while processing the chat request: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return jsonify({"reply": "Sorry, something went wrong. Please try again later."})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
