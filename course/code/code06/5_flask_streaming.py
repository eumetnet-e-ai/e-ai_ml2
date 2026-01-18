from flask import Flask, request, jsonify, render_template, Response
import ollama
import json

app = Flask(__name__)

chat_history = []  # Stores chat history
MAX_HISTORY = 20    # Keep only the last 5 exchanges

@app.route('/')
def index():
    return render_template('index_stream.html')  # Serves the UI

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    chat_history.append({"role": "user", "content": user_input})

    if len(chat_history) > MAX_HISTORY * 2:
        chat_history.pop(0)

    response_text = ""  # Placeholder for assistant response

    print("\n📌 DEBUG: Chat history before sending to Ollama:")
    print(json.dumps(chat_history, indent=2))

    def generate():
        nonlocal response_text
        response = ollama.chat(model="llama3", messages=chat_history, stream=True)

        for chunk in response:
            if "message" in chunk and "content" in chunk["message"]:
                text = chunk["message"]["content"]
                response_text += text
                yield f"data: {json.dumps({'text': text})}\n\n"

        # ✅ Store assistant response AFTER the full response is generated
        chat_history.append({"role": "assistant", "content": response_text})

        print("\n📌 DEBUG: Chat history AFTER response:")
        print(json.dumps(chat_history, indent=2))

    return Response(generate(), mimetype="text/event-stream")


if __name__ == '__main__':
    app.run(debug=True, threaded=True)  # Threaded mode for better performance
