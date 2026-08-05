from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ask_groq
import traceback

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "🏏 CricketSense AI Backend is running successfully!"
@app.route("/chat", methods=["POST"])
def chat():
    try:
        print("STEP 1")
        data = request.get_json()
        print("STEP 2")
        question = data.get("message")
        print("Question:", question)

        answer = ask_groq(question)
        print("STEP 2")

        print("Answer generated successfully")
        print("STEP 4")
        return jsonify({
            "reply": answer
        })
    except Exception as e:
        import traceback
        print("\n========== ERROR ==========")
        print("Exception:", repr(e))
        traceback.print_exc()
        print("===========================\n")
        return jsonify({
            "reply": str(e)
        }), 500
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)