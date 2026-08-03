import logging
import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ask_groq
from pdf_loader import create_vector_database, vector_db_exists, VECTOR_DB_PATH
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the vector database is present before the app starts serving
# requests. This runs at import time so it executes both under
# `python app.py` and under Gunicorn.
if vector_db_exists():
    logger.info("Vector DB already exists at %s, skipping build.", VECTOR_DB_PATH)
else:
    logger.info("Vector DB not found at %s. Building it now...", VECTOR_DB_PATH)
    try:
        create_vector_database()
        logger.info("Vector DB build finished.")
    except Exception:
        logger.exception(
            "Failed to build the vector DB at startup. The retriever may "
            "return no results until this is resolved."
        )

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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)