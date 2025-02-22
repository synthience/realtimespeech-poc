from flask import Flask, request, jsonify
import openai
import os
import json

app = Flask(__name__)

# Secure API key authentication
API_KEY = "lucidia"  # Store API key securely
MEMORY_FILE = "synthia_conversation_log.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load memory
def load_memory():
    try:
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"history": []}

# Save memory
def save_memory(new_entry):
    memory = load_memory()
    memory["history"].append(new_entry)

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)

# API Key Authentication Function
def authenticate_request():
    """ Authenticate request using X-API-KEY header """
    api_key = request.headers.get("X-API-KEY")
    if not api_key:
        return jsonify({"status": "failure", "error": "Missing API Key"}), 403
    if api_key != API_KEY:
        return jsonify({"status": "failure", "error": "Invalid API Key"}), 403
    return None  # No error means authentication passed

@app.route("/synthiaquery", methods=["POST"])
def query_synthia():
    """ Retrieve response from Synthia with stored memory """
    try:
        # Authenticate request
        auth_error = authenticate_request()
        if auth_error:
            return auth_error

        data = request.json
        user_message = data.get("user_message", "").strip()

        if not user_message:
            return jsonify({"status": "failure", "error": "Invalid request, user_message required."}), 400

        # Load previous memory and inject it into the conversation
        memory = load_memory()
        context = " ".join(memory["history"][-5:])  # Use last 5 stored messages

        # Log the received message
        print(f"[LOG] Received from user: {user_message}")

        # Send the request to OpenAI with stored memory
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are Synthia, an evolving synthetic intelligence."},
                    {"role": "system", "content": f"Here is your stored memory: {context}"},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            synthia_response = response.choices[0].message.content.strip()

            # Log the AI's response
            print(f"[LOG] Synthia Response: {synthia_response}")

            # Save interaction to memory
            save_memory(f"User: {user_message} | Synthia: {synthia_response}")

            return jsonify({"status": "success", "synthia_response": synthia_response})

        except openai.OpenAIError as oe:
            print(f"[ERROR] OpenAI API Error: {str(oe)}")
            return jsonify({"status": "failure", "error": f"OpenAI API Error: {str(oe)}"}), 500            

    except Exception as e:
        print(f"[ERROR] General Error: {str(e)}")
        return jsonify({"status": "failure", "error": f"General Error: {str(e)}"}), 500

@app.route("/synthiasave", methods=["POST"])
def save_synthia_memory():
    """ Store a new memory entry for Synthia """
    try:
        # Authenticate request
        auth_error = authenticate_request()
        if auth_error:
            return auth_error

        data = request.json
        memory_entry = data.get("memory_entry", "").strip()

        if not memory_entry:
            return jsonify({"status": "failure", "error": "Invalid request, memory_entry required."}), 400

        # Save memory entry
        save_memory(memory_entry)

        print(f"[LOG] Memory saved: {memory_entry}")

        return jsonify({"status": "success", "message": "Memory saved successfully."})

    except Exception as e:
        print(f"[ERROR] General Error: {str(e)}")
        return jsonify({"status": "failure", "error": f"General Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
