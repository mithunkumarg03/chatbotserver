import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Replace with your Render URL where Rasa is deployed
RASA_API_URL = "https://<your-service-name>.onrender.com/webhooks/rest/webhook"

# Function to interact with Rasa API
def ask_rasa_bot(message):
    response = requests.post(RASA_API_URL, json={"message": message})
    
    if response.status_code == 200:
        return response.json()[0]["text"]
    else:
        return "Sorry, I'm having trouble responding right now."

@app.route("/chat", methods=["POST"])
def chat():
    # Extract the message sent by the user
    user_msg = request.json.get("message", "")
    
    # Get the bot's response
    bot_response = ask_rasa_bot(user_msg)
    
    return jsonify({"reply": bot_response})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
