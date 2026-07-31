from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ask_groq
import traceback

app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        question = data.get("message")
        print("Question:", question)

        answer = ask_groq(question)

        print("Answer generated successfully")

        return jsonify({
            "reply": answer
        })

    except Exception as e:
        print("\n========== ERROR ==========")
        traceback.print_exc()
        print("===========================\n")

        return jsonify({
            "reply": "Server Error"
        }), 500

if __name__ == "__main__":
    app.run(debug=True)