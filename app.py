from flask import Flask, request, jsonify
import os
import requests
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 👉 Set your Groq API key in Render env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get("message", "")

        if not message:
            return jsonify({"reply": "No message received"}), 400

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "user", "content": message}
                ]
            }
        )

        result = response.json()

        print("FULL GROQ RESPONSE:", result)   # 🔥 DEBUG

        # ✅ SAFE EXTRACTION
        if "choices" in result:
            reply = result["choices"][0]["message"]["content"]
        else:
            return jsonify({
                "reply": f"Groq error: {result}"
            }), 200

        return jsonify({"reply": reply})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": "Server error"}), 500

@app.route('/')
def home():
    return 'Chatbot backend (Groq) is running.'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
