from flask import Flask, request, jsonify
from flask_cors import CORS
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

app = Flask(__name__)
CORS(app)

# Create chatbot instance
chatbot = ChatBot("MyBot", logic_adapters=[
    "chatterbot.logic.BestMatch"
])

# Train the chatbot (basic English)
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    response = chatbot.get_response(user_msg)
    return jsonify({"reply": str(response)})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server is Running!"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
