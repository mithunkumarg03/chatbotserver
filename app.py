from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow all origins for now (okay for dev)

# Set up API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load Chat-Bison (Gemini) model
model = genai.GenerativeModel("gemini-2.0-flash")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    
    if not message:
        return jsonify({"reply": "No message received."}), 400

    try:
        response = model.generate_content(message)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"}), 500

@app.route('/')
def home():
    return 'Chatbot backend is running.'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
