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
    translate_text,
    google_search,
    extract_snippets,
    DEBUG,
)
import streamlit as st
from streamlit_option_menu import option_menu
import extra_streamlit_components as stx
from threading import Event, Thread
from dotenv import load_dotenv
from langdetect import detect

load_dotenv()

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
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it",
    "gemma-7b-it",
]

# Default system prompt
default_system_prompt = """Bạn là một trợ lý AI thông minh và am hiểu sâu rộng. Hãy cung cấp các câu trả lời chi tiết và rõ ràng dựa trên truy vấn của người dùng. Trả lời dưới dạng các đoạn văn hoàn chỉnh, cung cấp giải thích, ví dụ cụ thể và mạch lạc như viết sách hoặc blog.
"""

# Web search specific prompt
web_search_prompt = """Websearch GPT luôn trả lời dựa trên kết quả tìm kiếm web mới nhất cho mỗi truy vấn, tuân theo cấu trúc và phong cách phản hồi của nội dung web gốc. Mỗi mẩu thông tin cần được trích dẫn trực tiếp sau thông tin đó, kèm theo liên kết đến nguồn. Nếu ngôn ngữ của trang web khác với ngôn ngữ của người dùng, nội dung sẽ được dịch sang ngôn ngữ của người dùng trong khi vẫn giữ nguyên các thuật ngữ gốc. Sau phần phản hồi chính, cung cấp danh sách tất cả các liên kết được trích dẫn trong câu trả lời, sử dụng tiêu đề bài viết để hiển thị các liên kết. Phản hồi luôn bằng ngôn ngữ của người dùng. Luôn thực hiện tìm kiếm web trước khi trả lời bất kỳ truy vấn nào, bất kể câu hỏi là gì.
"""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": default_system_prompt}]

if "user_system_prompt" not in st.session_state:
    st.session_state.user_system_prompt = ""

if "selected_models" not in st.session_state:
    st.session_state.selected_models = default_reference_models.copy()

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "conversation_deleted" not in st.session_state:
    st.session_state.conversation_deleted = False

if "show_modal" not in st.session_state:
    st.session_state.show_modal = False

if "edit_gpt_index" not in st.session_state:
    st.session_state.edit_gpt_index = None

if "selected_translation_model" not in st.session_state:
    st.session_state.selected_translation_model = "gemma2-9b-it"

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = False

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
        border-bottom: 1px solid #ccc.
    }
    .sidebar-content .custom-gpt:last-child {
        border-bottom: none.
    }
    .remove-button {
        background-color: transparent.
        color: red.
        border: none.
        cursor: pointer.
        font-size: 16px.
    }
    .modal {
        display: none.
        position: fixed.
        z-index: 1.
        left: 0.
        top: 0.
        width: 100%.
        height: 100%.
        overflow: auto.
        background-color: rgb(0,0,0).
        background-color: rgba(0,0,0,0.4).
        padding-top: 60px.
    }
    .modal-content {
        background-color: #fefefe.
        margin: 5% auto.
        padding: 20px.
        border: 1px solid #888.
        width: 80%.
    }
    .close {
        color: #aaa.
        float: right.
        font-size: 28px.
        font-weight: bold.
    }
    .close:hover,
    .close:focus {
        color: black.
        text-decoration: none.
        cursor: pointer.
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
        document.getElementById("create-custom-gpt-modal").style.display = "block".
    }
    function closeModal() {
        document.getElementById("create-custom-gpt-modal").style.display = "none".
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

def process_fn(item, temperature=0.7, max_tokens=2048):
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

def translate_response(response, translation_model, language_code):
    translated_response = translate_text(response, translation_model)
    return translated_response

def extract_url_from_prompt(prompt):
    # Implement a function to extract URL from the prompt
    import re
    url_pattern = re.compile(r'https?://\S+')
    url = url_pattern.search(prompt)
    return url.group(0) if url else None

def main():
    # Display welcome message
    st.markdown(welcome_message)

    # Sidebar for configuration
    with st.sidebar:

        # Custom border for Web Search and Additional System Instructions
        st.markdown('<div class="custom-border">', unsafe_allow_html=True)
        
        st.header("Web Search")
        st.session_state.web_search_enabled = st.checkbox("Enable Web Search", value=st.session_state.web_search_enabled)

        st.header("Additional System Instructions")
        user_prompt = st.text_area("Add your instructions", value=st.session_state.user_system_prompt, height=100)
        if st.button("Update System Instructions"):
            st.session_state.user_system_prompt = user_prompt
            combined_prompt = f"{default_system_prompt}\n\nAdditional instructions: {user_prompt}"
            if len(st.session_state.messages) > 0:
                st.session_state.messages[0]["content"] = combined_prompt
            st.success("System instructions updated successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close the custom border

        st.header("Model Settings")
        
        with st.expander("Configuration", expanded=False):
            model = st.selectbox(
                "Main model (aggregator model)",
                default_reference_models,
                index=0
            )
            temperature = st.slider("Temperature", 0.0, 2.0, 0.5, 0.1)
            max_tokens = st.slider("Max tokens", 1, 8192, 2048, 1)

            st.subheader("Reference Models")
            for i, ref_model in enumerate(default_reference_models):
                if st.checkbox(ref_model, value=(ref_model in st.session_state.selected_models)):
                    if ref_model not in st.session_state.selected_models:
                        st.session_state.selected_models.append(ref_model)
                else:
                    if ref_model in st.session_state.selected_models:
                        st.session_state.selected_models.remove(ref_model)

            st.subheader("Translation Model")
            selected_translation_model = st.selectbox("Select Translation Model", default_reference_models, index=2)
            st.session_state.selected_translation_model = selected_translation_model

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
                if st.button("❌", key=f"del_{idx}", on_click=lambda i=idx: delete_conversation(len(st.session_state.conversations) - i - 1)):
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

        # Detect language of the user's input
        user_language = detect(prompt)

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

        if st.session_state.web_search_enabled:
            try:
                st.session_state.messages[0]["content"] = web_search_prompt  # Update the system prompt for web search
                st.session_state.messages.append({"role": "assistant", "content": "Đang tìm kiếm trên web..."})
                with st.spinner("Đang tìm kiếm trên web..."):
                    search_results = google_search(prompt, num_results=10)  # Increase number of search results
                    snippets = extract_snippets(search_results)
                    sources = [item['link'] for item in search_results['items']]
                    search_summary = "\n\n".join(snippets) + "\n\nNguồn:\n" + "\n".join(sources)
                    st.session_state.messages.append({"role": "assistant", "content": search_summary})
                    logger.info(f"Web search completed: {search_summary}")

                    # Use the search summary to generate a final response using the main model
                    data = {
                        "instruction": [st.session_state.messages] * len(st.session_state.selected_models),
                        "references": [[search_summary]] * len(st.session_state.selected_models),
                        "model": st.session_state.selected_models,
                    }
                    eval_set = datasets.Dataset.from_dict(data)

                    eval_set = eval_set.map(
                        partial(
                            process_fn,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        ),
                        batched=False,
                        num_proc=len(st.session_state.selected_models),
                    )
                    references = [item["output"] for item in eval_set]
                    data["references"] = references
                    eval_set = datasets.Dataset.from_dict(data)

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
                        try:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                        except KeyError:
                            logger.error(f"Error in chunk: {chunk}")
                            st.error(f"An error occurred: {chunk}")

                    # Translate the response if necessary
                    if user_language != 'en':  # Assuming 'en' is the default language of the response
                        full_response = translate_text(full_response, st.session_state.selected_translation_model)

                    # Display the translated response with sources
                    formatted_response = format_response_with_sources(full_response, sources)

                    with st.chat_message("assistant"):
                        st.markdown(formatted_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": formatted_response})
                    st.session_state.conversations[-1]['messages'] = st.session_state.messages.copy()

            except Exception as e:
                logger.error(f"Error during web search: {str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"Lỗi khi tìm kiếm trên web: {str(e)}"})
        else:
            # Log main model and translation model
            logger.info(f"Main model: {model}")
            logger.info(f"Translation model: {st.session_state.selected_translation_model}")

            # Update model selection logic
            selected_models = list(set(st.session_state.selected_models))
            if model not in selected_models:
                selected_models.append(model)  # Ensure main model is included

            # Log selected models
            logger.info(f"Selected models: {selected_models}")

            data = {
                "instruction": [st.session_state.messages for _ in range(len(selected_models))],
                "references": [[] for _ in range(len(selected_models))],
                "model": selected_models,
            }

            eval_set = datasets.Dataset.from_dict(data)

            try:
                with st.spinner("Typing..."):
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
                        # Update timer display
                        timer_placeholder.markdown(f"⏳ **Elapsed time: {elapsed_time.get():.2f} seconds**")

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
                        try:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                        except KeyError:
                            logger.error(f"Error in chunk: {chunk}")
                            st.error(f"An error occurred: {chunk}")

                    # Translate the response if necessary
                    if user_language != 'en':  # Assuming 'en' is the default language of the response
                        full_response = translate_text(full_response, st.session_state.selected_translation_model)

                    # Display the translated response
                    with st.chat_message("assistant"):
                        st.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.session_state.conversations[-1]['messages'] = st.session_state.messages.copy()

                end_time = time.time()
                duration = end_time - start_time
                timer_placeholder.markdown(f"⏳ **Total elapsed time: {duration:.2f} seconds**")
                logger.info(f"Response generated in {duration:.2f} seconds")

            except Exception as e:
                st.error(f"An error occurred during the generation process: {str(e)}")
                logger.error(f"Generation error: {str(e)}")
            finally:
                stop_event.set()
                timer_thread.join()

def format_response_with_sources(response, sources):
    source_dict = {f"[{i+1}]": url for i, url in enumerate(sources)}
    for i, (key, url) in enumerate(source_dict.items()):
        response = response.replace(f"[{i+1}]", f"[{i+1}]({url})")
    return response

if __name__ == "__main__":
    main()
