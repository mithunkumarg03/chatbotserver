from flask import Flask, request, jsonify
import google.generativeai as genai
from pymongo import MongoClient
from bson import ObjectId
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
import os
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# --- JWT Setup ---
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
jwt = JWTManager(app)
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1) 
jwt = JWTManager(app)

# --- MongoDB Connection ---
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client.get_default_database()
chats_col = db["chats"]

# --- Gemini Setup ---
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# =====================================================
# POST /chat → generate reply using chat history in MongoDB
# =====================================================
@app.route("/chat", methods=["POST"])
@jwt_required()
def chat():
    user_id = get_jwt_identity()
    data = request.get_json()
    message = data.get("message")
    chat_id = data.get("chat_id")

    if not message:
        return jsonify({"error": "Message is required"}), 400

    # --- Fetch history from MongoDB ---
    history = []
    if chat_id:
        chat_doc = chats_col.find_one({
            "_id": ObjectId(chat_id),
            "user_id": ObjectId(user_id)
        })
        if chat_doc:
            # Convert MongoDB messages into Gemini history format
            history = [
                {"role": msg["role"], "parts": [msg["content"]]}
                for msg in chat_doc.get("messages", [])
            ]

    # --- Create Gemini chat with existing history ---
    chat = model.start_chat(history=history)
    response = chat.send_message(message)

    # ✅ Return only the bot reply (frontend will handle saving)
    return jsonify({"reply": response.text}), 200


@app.route("/")
def home():
    return "Gemini chatbot backend (context-aware, MongoDB-connected)."


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
