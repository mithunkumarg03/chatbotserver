from flask import Flask, request, jsonify
from deeppavlov import build_model, configs

# Initialize the DeepPavlov model (e.g., a pre-trained model for a conversational bot)
model = build_model(configs.squad.squad_bert, download=True)

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "")
    
    # DeepPavlov will generate a response to the user input
    response = model([user_msg])
    
    return jsonify({"reply": response[0]})

@app.route("/", methods=["GET"])
def home():
    return "Chatbot Server Running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
