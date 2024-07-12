import os
import json
import datasets
import threading
import time
from functools import partial
from loguru import logger
from utils import (
    generate_together_stream,
    generate_with_references,
    DEBUG,
)
from datasets.utils.logging import disable_progress_bar
import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from threading import Event, Thread
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import GoogleAPIError

load_dotenv()

# Path to your Firebase Admin SDK private key file
cred_path = r'moa-groq-firebase-adminsdk-qo6gc-9259e672b1.json'
if not os.path.exists(cred_path):
    raise FileNotFoundError(f"Credential file not found: {cred_path}")

cred = credentials.Certificate(cred_path)

# Initialize Firebase app if not already initialized
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

class SharedValue:
    def __init__(self, initial_value=0.0):
        self.value = initial_value
        self.lock = threading.Lock()

    def set(self, new_value):
        with self.lock:
            self.value = new_value

    def get(self):
        with self.lock:
            return self.value

# Default reference models
default_reference_models = [
    "gemma2-9b-it",
    "gemma-7b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
]

# Default system prompt
default_system_prompt = """You are an AI assistant named MoA, powered by a Mixture of Agents architecture. 
Your role is to provide helpful, accurate, and ethical responses to user queries. 
You have access to multiple language models and can leverage their combined knowledge to generate comprehensive answers. 
Always strive to be respectful, avoid harmful content, and admit when you're unsure about something.
Always respond in the user's language."""

# User data management functions
def save_user_data(email):
    if email:  # Ensure email is set before saving data
        user_data = {
            "messages": st.session_state.messages,
            "user_system_prompt": st.session_state.user_system_prompt,
            "selected_models": st.session_state.selected_models,
            "conversations": st.session_state.conversations,
            "custom_gpts": st.session_state.custom_gpts,
        }
        logger.info(f"Saving user data for {email}")
        try:
            db.collection('user_data').document(email).set(user_data)
        except GoogleAPIError as e:
            logger.error(f"Failed to save user data for {email}: {e}")
            st.error("Failed to save user data. Please try again later.")

def load_user_data(email):
    if email:  # Ensure email is set before loading data
        doc_ref = db.collection('user_data').document(email)
        try:
            doc = doc_ref.get()
            if doc.exists:
                logger.info(f"Loading user data for {email}")
                user_data = doc.to_dict()
                st.session_state.messages = user_data.get("messages", [{"role": "system", "content": default_system_prompt}])
                st.session_state.user_system_prompt = user_data.get("user_system_prompt", "")
                st.session_state.selected_models = user_data.get("selected_models", default_reference_models.copy())
                st.session_state.conversations = user_data.get("conversations", [])
                st.session_state.custom_gpts = user_data.get("custom_gpts", [])
            else:
                logger.info(f"No existing user data for {email}, initializing defaults")
                st.session_state.messages = [{"role": "system", "content": default_system_prompt}]
                st.session_state.user_system_prompt = ""
                st.session_state.selected_models = default_reference_models.copy()
                st.session_state.conversations = []
                st.session_state.custom_gpts = []
        except GoogleAPIError as e:
            logger.error(f"Failed to load user data for {email}: {e}")
            st.error("Failed to load user data. Please try again later.")

def delete_conversation(index):
    st.session_state.conversations.pop(index)
    if st.session_state.user_email:  # Ensure email is set before saving data
        save_user_data(st.session_state.user_email)
    st.session_state.conversation_deleted = True

# Initialize session state
if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": default_system_prompt}]

if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = ""

if "selected_models" not in st.session_state:
    st.session_state.selected_models = default_reference_models.copy()

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "custom_gpts" not in st.session_state:
    st.session_state.custom_gpts = []

if "conversation_deleted" not in st.session_state:
    st.session_state.conversation_deleted = False

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

if "edit_gpt_index" not in st.session_state:
    st.session_state.edit_gpt_index = None

disable_progress_bar()

# Set page configuration
st.set_page_config(page_title="Groq MoA Chatbot", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    .sidebar-content {
        padding: 1rem;
    }
    .sidebar-content .custom-gpt {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem;
        border-bottom: 1px solid #ccc;
    }
    .sidebar-content .custom-gpt:last-child {
        border-bottom: none;
    }
    .remove-button {
        background-color: transparent;
        color: red;
        border: none;
        cursor: pointer;
        font-size: 16px;
    }
    .modal {
        display: none;
        position: fixed;
        z-index: 1;
        left: 0;
        top: 0;
        width: 100%;
        height: 100%;
        overflow: auto;
        background-color: rgb(0,0,0);
        background-color: rgba(0,0,0,0.4);
        padding-top: 60px;
    }
    .modal-content {
        background-color: #fefefe;
        margin: 5% auto;
        padding: 20px;
        border: 1px solid #888;
        width: 80%;
    }
    .close {
        color: #aaa;
        float: right;
        font-size: 28px;
        font-weight: bold;
    }
    .close:hover,
    .close:focus {
        color: black;
        text-decoration: none;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# JavaScript to handle modal display
st.markdown(
    """
    <script>
    function openModal() {
        document.getElementById("create-custom-gpt-modal").style.display = "block";
    }
    function closeModal() {
        document.getElementById("create-custom-gpt-modal").style.display = "none";
    }
    </script>
    """,
    unsafe_allow_html=True
)

# Welcome message
welcome_message = """
# MoA (Mixture-of-Agents) Chatbot

Phương pháp Mixture of Agents (MoA) là một kỹ thuật mới, tổ chức nhiều mô hình ngôn ngữ lớn (LLM) thành một kiến trúc nhiều lớp. Mỗi lớp bao gồm nhiều tác nhân (mô hình LLM riêng lẻ). Các tác nhân này hợp tác với nhau bằng cách tạo ra các phản hồi dựa trên đầu ra từ các tác nhân ở lớp trước, từng bước tinh chỉnh và cải thiện kết quả cuối cùng, chỉ sử dụng các mô hình mã nguồn mở (Open-source)!

Truy cập Bài nghiên cứu gốc để biết thêm chi tiết [Mixture-of-Agents Enhances Large Language Model Capabilities](https://arxiv.org/abs/2406.04692)

Chatbot này sử dụng các mô hình ngôn ngữ lớn (LLM) sau đây làm các lớp – Mô hình tham chiếu, sau đó chuyển kết quả cho mô hình tổng hợp để tạo ra phản hồi cuối cùng.
"""

def process_fn(item, temperature=0.5, max_tokens=8192):
    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if DEBUG:
        logger.info(
            f"model {model}, instruction {item['instruction']}, output {output[:20]}",
        )

    st.write(f"Finished querying {model}.")

    return {"output": output}

def run_timer(stop_event, elapsed_time):
    start_time = time.time()
    while not stop_event.is_set():
        elapsed_time.set(time.time() - start_time)
        time.sleep(0.1)

def main():
    # Display welcome message
    st.markdown(welcome_message)
    
    # Login system
    if st.session_state.user_email is None:
        st.sidebar.subheader("Login")
        email = st.sidebar.text_input("Email")
        if st.sidebar.button("Login"):
            logger.info(f"User {email} is logging in")
            st.session_state.user_email = email
            load_user_data(email)
            st.rerun()
    else:
        st.sidebar.markdown(f"Welcome, {st.session_state.user_email}!")
        if st.sidebar.button("Logout"):
            logger.info(f"User {st.session_state.user_email} is logging out")
            save_user_data(st.session_state.user_email)
            st.session_state.user_email = None
            st.rerun()

    # Sidebar for configuration
    with st.sidebar:
        st.sidebar.header("Settings")
        
        with st.expander("Configuration", expanded=False):
            model = st.selectbox(
                "Main model (aggregator model)",
                default_reference_models,
                index=0
            )
            temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
            max_tokens = st.slider("Max tokens", 1, 8192, 8192, 1)

            st.subheader("Reference Models")
            for i, ref_model in enumerate(default_reference_models):
                if st.checkbox(ref_model, value=(ref_model in st.session_state.selected_models)):
                    if ref_model not in st.session_state.selected_models:
                        st.session_state.selected_models.append(ref_model)
                else:
                    if ref_model in st.session_state.selected_models:
                        st.session_state.selected_models.remove(ref_model)

            st.subheader("Additional System Instructions")
            user_prompt = st.text_area("Add your instructions", value=st.session_state.user_system_prompt, height=100)

            if st.button("Update System Instructions"):
                st.session_state.user_system_prompt = user_prompt
                combined_prompt = f"{default_system_prompt}\n\nAdditional instructions: {user_prompt}"
                if len(st.session_state.messages) > 0:
                    st.session_state.messages[0]["content"] = combined_prompt
                st.success("System instructions updated successfully!")

        # Custom GPT creation button
        st.button("Create Custom GPT", on_click=lambda: st.session_state.update({"show_modal": True}))

        # Modal window for creating custom GPT
        if st.session_state.get('show_modal', False):
            st.markdown(
                """
                <div id="create-custom-gpt-modal" class="modal" style="display:block;">
                  <div class="modal-content">
                    <span class="close" onclick="closeModal();">&times;</span>
                    <h2>Create Custom GPT</h2>
                    <form id="create-gpt-form">
                      <label for="gpt-name">Name your GPT:</label>
                      <input type="text" id="gpt-name" name="gpt-name" required>
                      <label for="gpt-instructions">Instructions for your GPT:</label>
                      <textarea id="gpt-instructions" name="gpt-instructions" rows="5" required></textarea>
                      <div class="modal-buttons">
                        <button type="button" onclick="submitGptForm();">Create</button>
                        <button type="button" class="cancel-btn" onclick="closeModal();">Cancel</button>
                      </div>
                    </form>
                  </div>
                </div>
                <script>
                function closeModal() {
                    document.getElementById("create-custom-gpt-modal").style.display = "none";
                    fetch('/?close_modal=true');
                }
                function submitGptForm() {
                    var name = document.getElementById("gpt-name").value;
                    var instructions = document.getElementById("gpt-instructions").value;
                    fetch('/?create_gpt=true&name=' + encodeURIComponent(name) + '&instructions=' + encodeURIComponent(instructions))
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            closeModal();
                            window.location.reload();
                        } else {
                            alert('Failed to create GPT');
                        }
                    });
                }
                </script>
                """,
                unsafe_allow_html=True
            )

        # Handle modal actions
        query_params = st.query_params
        if "close_modal" in query_params:
            st.session_state.show_modal = False
            st.set_query_params()

        if "create_gpt" in query_params:
            name = query_params.get("name")[0]
            instructions = query_params.get("instructions")[0]
            new_gpt = {"name": name, "instructions": instructions}
            st.session_state.custom_gpts.append(new_gpt)
            save_user_data(st.session_state.user_email)
            st.set_query_params(create_gpt=None)
            st.experimental_rerun()

        # List of custom GPTs
        st.sidebar.subheader("Your Custom GPTs")
        for idx, gpt in enumerate(st.session_state.custom_gpts):
            with st.sidebar:
                gpt_container = st.container()
                with gpt_container:
                    if st.button(gpt['name'], key=f"view_gpt_{idx}"):
                        st.write(f"**Name:** {gpt['name']}")
                        st.write(f"**Instructions:** {gpt['instructions']}")
                        st.write(f"**Created At:** {gpt.get('created_at', 'N/A')}")
                        if st.button("Edit", key=f"edit_gpt_{idx}"):
                            st.session_state.edit_gpt_index = idx
                            st.session_state.show_modal = True
                        if st.button("Delete", key=f"delete_gpt_{idx}"):
                            st.session_state.custom_gpts.pop(idx)
                            if st.session_state.user_email:
                                save_user_data(st.session_state.user_email)
                            st.experimental_rerun()

        # Start new conversation button
        if st.button("Start New Conversation", key="new_conversation"):
            st.session_state.messages = [{"role": "system", "content": st.session_state.messages[0]["content"]}]
            st.rerun()

        # Previous conversations
        st.subheader("Previous Conversations")
        for idx, conv in enumerate(reversed(st.session_state.conversations)):  # Reverse the list
            cols = st.columns([0.9, 0.1])
            with cols[0]:
                if st.button(f"{len(st.session_state.conversations) - idx}. {conv['first_question'][:30]}...", key=f"conv_{idx}"):
                    st.session_state.messages = conv['messages']
                    st.rerun()
            with cols[1]:
                if st.button("❌", key=f"del_{idx}", on_click=delete_conversation, args=(len(st.session_state.conversations) - idx - 1,)):
                    st.session_state.conversation_deleted = True

        # Add a download button for chat history
        if st.button("Download Chat History"):
            chat_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[1:]])  # Skip system message
            st.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain"
            )

    # Trigger rerun if a conversation was deleted
    if st.session_state.conversation_deleted:
        st.session_state.conversation_deleted = False
        st.experimental_rerun()

    # Chat interface
    st.header("💬 Chat with MoA")
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages[1:]:  # Skip the system message
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What would you like to know?"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Save first question of new conversation
        if len(st.session_state.messages) == 2:  # First user message
            st.session_state.conversations.append({
                "first_question": prompt,
                "messages": st.session_state.messages.copy()
            })

        # Generate response
        timer_placeholder = st.empty()
        stop_event = threading.Event()
        elapsed_time = SharedValue()
        timer_thread = threading.Thread(target=run_timer, args=(stop_event, elapsed_time))
        timer_thread.start()

        start_time = time.time()

        # Update model selection logic
        selected_models = list(set(st.session_state.selected_models) - set([model]))
        if not selected_models:
            selected_models = [model]  # Use main model if no other models are selected

        data = {
            "instruction": [st.session_state.messages for _ in range(len(selected_models))],
            "references": [[] for _ in range(len(selected_models))],
            "model": selected_models,
        }

        eval_set = datasets.Dataset.from_dict(data)

        try:
            with st.spinner("Thinking..."):
                progress_bar = st.progress(0)
                for i_round in range(1):
                    eval_set = eval_set.map(
                        partial(
                            process_fn,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        batched=False,
                        num_proc=len(selected_models),
                    )
                    references = [item["output"] for item in eval_set]
                    data["references"] = references
                    eval_set = datasets.Dataset.from_dict(data)
                    progress_bar.progress((i_round + 1) / 1)
                    # Update timer display
                    timer_placeholder.markdown(f"⏳ **Elapsed time: {elapsed_time.get():.2f} seconds**")

                st.write("Aggregating results & querying the aggregate model...")
                output = generate_with_references(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=st.session_state.messages,
                    references=references,
                    generate_fn=generate_together_stream
                )

                full_response = ""
                for chunk in output:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content

                with st.chat_message("assistant"):
                    st.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.session_state.conversations[-1]['messages'] = st.session_state.messages.copy()
                if st.session_state.user_email:
                    save_user_data(st.session_state.user_email)  # Save data after the response

            end_time = time.time()
            duration = end_time - start_time
            timer_placeholder.markdown(f"⏳ **Total elapsed time: {duration:.2f} seconds**")

        except Exception as e:
            st.error(f"An error occurred during the generation process: {str(e)}")
            logger.error(f"Generation error: {str(e)}")
        finally:
            stop_event.set()
            timer_thread.join()

    # Auto-save user data after each interaction
    if st.session_state.user_email:
        save_user_data(st.session_state.user_email)

if __name__ == "__main__":
    main()
