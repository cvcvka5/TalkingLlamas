from flask import Flask, request, jsonify
from mods.llama import Llama
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

system_prompt = Llama.generateMessage(
    "You are my assistant, Kyle. You must answer really briefly with short sentences. Your responses must not be longer than 10 words.", role="system"
)
llama_instance = Llama([system_prompt], name="API_Llama")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle a chat request by processing the user's message and returning a response.
    This function expects a JSON payload in the request body with a "message" key.
    It sends the user's message to the `llama_instance` for processing and returns
    the generated response along with the message history.
    Returns:
        Response: A JSON response containing:
            - "response" (str): The generated response from `llama_instance`.
            - "history" (list): The message history excluding the initial system message.
        If the "message" key is missing in the request body, returns a 400 error with
        an appropriate error message.
        If an exception occurs during processing, returns a 500 error with the exception message.
    Raises:
        Exception: If an error occurs while sending the message to `llama_instance`.
    """
    
    data = request.get_json()
    
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' in request body"}), 400

    user_message = data["message"]
    
    try:
        response = llama_instance.send(user_message)
        return jsonify({
            "response": response,
            "history": llama_instance.messages[1:]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["delete"])
def reset():
    """
    Resets the conversation by clearing the current messages and reinitializing 
    them with the system prompt.
    Returns:
        Response: A JSON response indicating the status of the reset operation.
    """
    
    llama_instance.messages = [system_prompt]
    return jsonify({"status": "conversation reset"})


if __name__ == "__main__":
    app.run(port=1456)
    
    
    
# Example
# response = r.post("http://127.0.0.1:1456/chat", json={"message": "whats the origin of the cosmos"})
# print(response.json()["response"])
# response = r.post("http://127.0.0.1:1456/chat", json={"message": "whats the origin of the sea"})
# print(response.json()["response"])
# response = r.post("http://127.0.0.1:1456/chat", json={"message": "what was my first question?"})
# print(response.json()["response"])
# response = r.delete("http://127.0.0.1:1456/reset")
# print(response.json()["status"])
# response = r.post("http://127.0.0.1:1456/chat", json={"message": "what was my first question?"})
# print(response.json()["response"])