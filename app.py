from flask import Flask, request, jsonify
from chatterbot import ChatBot

app = Flask(__name__)

chatbot = ChatBot('DynamicBot')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get("message", "")
    response = chatbot.get_response(user_message)
    return jsonify({"reply": str(response)})


@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server is Running!"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
