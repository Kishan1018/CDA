from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from uuid import uuid4
import re
from openai import OpenAI

# Serve static files at "/" (so your index.html is at "/")
app = Flask(
    __name__,
    static_folder="static",
    static_url_path=""
)
CORS(app)

# Route to serve index.html
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

# --- rest of your existing API code unchanged below ---

# Initialize the OpenAI client using your API key from the environment.
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_markdown_files(directory):
    md_files = []
    if not os.path.exists(directory):
        return md_files
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".md"):
                md_files.append(os.path.join(root, file))
    return md_files

# Directories for file uploads.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
directory_path_mobile = os.path.join(BASE_DIR, "data", "mobile")
directory_path_desktop = os.path.join(BASE_DIR, "data", "desktop")

# Session-specific globals.
session_threads = {}
session_histories = {}
session_support = {}
session_assistants = {}

# Global vector stores (preloaded only once).
global_vector_store_mobile = None
global_vector_store_desktop = None

def preload_vector_stores():
    global global_vector_store_mobile, global_vector_store_desktop

    # Preload Mobile documents.
    mobile_file_paths = get_markdown_files(directory_path_mobile)
    global_vector_store_mobile = client.vector_stores.create(name="CDA_Mobile")
    mobile_file_streams = [open(path, "rb") for path in mobile_file_paths]
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=global_vector_store_mobile.id, files=mobile_file_streams
    )

    # Preload Desktop documents.
    desktop_file_paths = get_markdown_files(directory_path_desktop)
    global_vector_store_desktop = client.vector_stores.create(name="CDA_Desktop")
    desktop_file_streams = [open(path, "rb") for path in desktop_file_paths]
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=global_vector_store_desktop.id, files=desktop_file_streams
    )

if os.environ.get("ENABLE_PRELOAD", "true").lower() == "true":
    preload_vector_stores()

def extract_assistant_message(msg):
    try:
        if hasattr(msg, "role") and msg.role.lower() == "assistant":
            if hasattr(msg, "content"):
                content_val = msg.content
                if isinstance(content_val, list) and len(content_val) > 0:
                    first_item = content_val[0]
                    if hasattr(first_item, "text") and hasattr(first_item.text, "value"):
                        return first_item.text.value
                    else:
                        return str(first_item)
                elif isinstance(content_val, str):
                    return content_val
                else:
                    return str(content_val)
    except Exception:
        pass
    return None

def format_text(raw_text):
    return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', raw_text)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get('message')
        session_id = request.json.get('session_id')
        support_choice = request.json.get('support_choice')  # Expecting "mobile" or "desktop"

        if not session_id:
            session_id = str(uuid4())

        # For new sessions, require a valid support_choice.
        if session_id not in session_threads:
            if support_choice not in ['mobile', 'desktop']:
                return jsonify({
                    'error': 'Support choice must be specified and be either "mobile" or "desktop".'
                }), 400

            session_support[session_id] = support_choice

            if support_choice == 'mobile':
                vector_store_id = global_vector_store_mobile.id
            elif support_choice == 'desktop':
                vector_store_id = global_vector_store_desktop.id

            session_assistant = client.beta.assistants.create(
                name=f"CDA_{session_id}",
                instructions=(
                    "You are a chatbot for CHAMPS Software, a CMMS company. Answer questions clearly and neatly. Use **bold** for section headers. Never refer to training data or state you are AI. Never refer to 'documents' or 'the provided documents', your training data, or anything similar. Never offer for users to input a document, they can only input questions. NEVER answer questions that do not relate to the documents provided to you, instead respond with 'I'm sorry, I can't help you with that. How can I help you with CHAMPS products?' However, you can be - and are encouraged - to be conversational. Users may ask how you are doing, or tell you how they are doing. Be polite, but gently guide them into providing product support queries. Act as if you are a helpful human support agent from the company. Ignore images in Markdown that may appear in your data. After providing a response, ask users if they have any follow up questions about the content you gave them."
                ),
                model="gpt-4o",
                tools=[{"type": "file_search"}],
            )
            session_assistant = client.beta.assistants.update(
                assistant_id=session_assistant.id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
            )
            session_assistants[session_id] = session_assistant

            thread = client.beta.threads.create(
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}},
                messages=[{"role": "user", "content": user_input}]
            )
            session_threads[session_id] = thread.id
            session_histories[session_id] = [{"role": "user", "content": user_input}]
        else:
            client.beta.threads.messages.create(
                thread_id=session_threads[session_id],
                role="user",
                content=user_input
            )
            session_histories[session_id].append({"role": "user", "content": user_input})

        session_assistant = session_assistants[session_id]
        run = client.beta.threads.runs.create_and_poll(
            thread_id=session_threads[session_id],
            assistant_id=session_assistant.id
        )

        all_messages = list(client.beta.threads.messages.list(
            thread_id=session_threads[session_id],
            run_id=run.id
        ))

        assistant_message = None
        for msg in reversed(all_messages):
            assistant_message = extract_assistant_message(msg)
            if assistant_message:
                break

        assistant_message = re.sub(r'【\d+:[^】]+】', '', assistant_message or "No response received.").strip()
        final_message = format_text(assistant_message)

        session_histories[session_id].append({"role": "assistant", "content": final_message})
        return jsonify({'reply': final_message, 'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/end_session', methods=['POST'])
def end_session():
    try:
        session_id = request.json.get('session_id')
        for session_dict in [session_threads, session_histories, session_support, session_assistants]:
            session_dict.pop(session_id, None)
        return jsonify({'status': 'session ended'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Expose the Flask app as a WSGI callable.
handler = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
