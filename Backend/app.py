from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import ask_groq
import traceback
import os
 
 
# ============================================================
# FLASK APP
# ============================================================
 
app = Flask(__name__)
 
# Allow frontend requests
CORS(app)
 
 
# ============================================================
# HOME ROUTE
# ============================================================
 
@app.route("/", methods=["GET"])
def home():
    return "🏏 CricketSense AI Backend is running successfully!"
 
 
# ============================================================
# CHAT ROUTE
# ============================================================
 
@app.route("/chat", methods=["POST"])
def chat():
 
    try:
 
        print("\n" + "=" * 60)
        print("NEW CHAT REQUEST")
        print("=" * 60)
 
        # ----------------------------------------------------
        # READ REQUEST
        # ----------------------------------------------------
 
        print("STEP 1: Reading request")
 
        data = request.get_json(silent=True)
 
        if not data:
            print("ERROR: No JSON data received")
 
            return jsonify({
                "reply": "No JSON data received."
            }), 400
 
        print("STEP 2: JSON received")
 
        # ----------------------------------------------------
        # GET QUESTION
        # ----------------------------------------------------
 
        question = data.get("message")
 
        if not isinstance(question, str) or not question.strip():
 
            print("ERROR: Empty question received")
 
            return jsonify({
                "reply": "Please provide a cricket-related question."
            }), 400
 
        question = question.strip()
 
        print("Question:", question)
 
        # ----------------------------------------------------
        # ASK CRICKETSENSE AI
        # ----------------------------------------------------
 
        print("STEP 3: Sending question to CricketSense AI")
 
        answer = ask_groq(question)
 
        print("STEP 4: Answer generated successfully")
 
        # ----------------------------------------------------
        # RETURN RESPONSE
        # ----------------------------------------------------
 
        print("STEP 5: Sending response to frontend")
 
        return jsonify({
            "reply": answer
        }), 200
 
 
    except Exception as e:
 
        print("\n" + "=" * 60)
        print("CRICKETSENSE AI BACKEND ERROR")
        print("=" * 60)
 
        print("Exception:", repr(e))
 
        traceback.print_exc()
 
        print("=" * 60 + "\n")
 
        return jsonify({
            "reply": "Sorry, CricketSense AI encountered a backend error."
        }), 500
 
 
# ============================================================
# RUN SERVER
# ============================================================
 
if __name__ == "__main__":
 
    port = int(os.environ.get("PORT", 5000))
 
    # debug=True must never run in production (Render) — it exposes the
    # Werkzeug interactive debugger and reloads on every file change.
    # Set FLASK_DEBUG=1 locally if you want it; Render won't set that,
    # so it defaults to False there automatically.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
 
    print("\n" + "=" * 60)
    print("🏏 CricketSense AI Backend Starting")
    print("=" * 60)
 
    print(f"Port          : {port}")
    print("Host          : 0.0.0.0")
    print(f"Debug mode    : {debug_mode}")
    print("Home endpoint : /")
    print("Chat endpoint : /chat")
    print("=" * 60)
 
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug_mode
    )
 