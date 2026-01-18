from flask import Flask, request, jsonify, render_template
import ollama

app = Flask(__name__, template_folder='templates')
chat_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400

    chat_history.append({"role": "user", "content": user_input})
    response = ollama.chat(model="llama3", messages=chat_history)
    reply = response['message']['content']
    chat_history.append({"role": "assistant", "content": reply})

    return jsonify({'response': reply})

if __name__ == '__main__':
    app.run(debug=True)
