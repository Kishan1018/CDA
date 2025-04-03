from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from uuid import uuid4
import re
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Initialize the OpenAI client using your API key from the environment
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

# Directories for file uploads
directory_path_mobile = '/data/all/'
directory_path_desktop = '/data/desktop/'
directory_path_all_CHAMPS = '/data/all/'

session_threads = {}
session_histories = {}
session_support = {}
session_assistants = {}

global_vector_store_mobile = None
global_vector_store_desktop = None
global_vector_store_all = None

def preload_vector_stores():
    global global_vector_store_mobile, global_vector_store_desktop, global_vector_store_all

    mobile_file_paths = get_markdown_files(directory_path_mobile)
    global_vector_store_mobile = client.vector_stores.create(name="CDA_Mobile")
    mobile_file_streams = [open(path, "rb") for path in mobile_file_paths]
    mobile_file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=global_vector_store_mobile.id, files=mobile_file_streams
    )

    desktop_file_paths = get_markdown_files(directory_path_desktop)
    global_vector_store_desktop = client.vector_stores.create(name="CDA_Desktop")
    desktop_file_streams = [open(path, "rb") for path in desktop_file_paths]
    desktop_file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=global_vector_store_desktop.id, files=desktop_file_streams
    )

    all_file_paths = get_markdown_files(directory_path_all_CHAMPS)
    global_vector_store_all = client.vector_stores.create(name="CDA_All")
    all_file_streams = [open(path, "rb") for path in all_file_paths]
    all_file_batch = client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=global_vector_store_all.id, files=all_file_streams
    )

# For testing on Vercel you might disable this preload (or control it via an environment variable)
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
        support_choice = request.json.get('support_choice')

        if not session_id:
            session_id = str(uuid4())

        if session_id not in session_threads:
            if support_choice not in ['mobile', 'desktop']:
                support_choice = 'all'
            session_support[session_id] = support_choice

            if support_choice == 'mobile':
                vector_store_id = global_vector_store_mobile.id
            elif support_choice == 'desktop':
                vector_store_id = global_vector_store_desktop.id
            else:
                vector_store_id = global_vector_store_all.id

            session_assistant = client.beta.assistants.create(
                name=f"CDA_{session_id}",
                instructions=(
                    "You are a chatbot for CHAMPS Software. Answer questions clearly and neatly. "
                    "Use **bold** for section headers. Never refer to training data or say you're AI. "
                    "Act as if you're a helpful human support agent from the company. Ignore images in Markdown."
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

# Expose the Flask app as a WSGI callable for Vercel
handler = app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
