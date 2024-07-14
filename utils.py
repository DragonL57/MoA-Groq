import os
import json
import time
import requests
import openai
import copy

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DEBUG = int(os.environ.get("DEBUG", "0"))

def generate_together(
    model,
    messages,
    max_tokens=6000,
    temperature=0.7,
    streaming=True,
):
    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            api_key = os.environ.get('GROQ_API_KEY')

            if api_key is None:
                logger.error("GROQ_API_KEY is not set")
                return None

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]
            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:
        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output

def generate_together_stream(
    model,
    messages,
    max_tokens=6000,
    temperature=0.7,
):
    endpoint = "https://api.groq.com/openai/v1"
    api_key = os.environ.get('GROQ_API_KEY')

    if api_key is None:
        logger.error("GROQ_API_KEY is not set")
        return None

    client = openai.OpenAI(
        api_key=api_key, base_url=endpoint
    )
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response

def generate_openai(
    model,
    messages,
    max_tokens=6000,
    temperature=0.7,
):
    api_key = os.environ.get('OPENAI_API_KEY')

    if api_key is None:
        logger.error("OPENAI_API_KEY is not set")
        return None

    client = openai.OpenAI(
        api_key=api_key,
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output

def translate_text(text, translation_model):
    api_key = os.environ.get('GROQ_API_KEY')

    if api_key is None:
        logger.error("GROQ_API_KEY is not set")
        return None

    prompt = {
        "role": "system",
        "content": "Hãy dịch chính xác phản hồi sau đây sang ngôn ngữ của người dùng, giữ nguyên các thuật ngữ chuyên ngành và đảm bảo rằng ý nghĩa và ngữ cảnh ban đầu được giữ nguyên. Đảm bảo rằng bản dịch rõ ràng, dễ hiểu và mạch lạc như cách viết sách hoặc blog."
    }

    messages = [
        prompt,
        {"role": "user", "content": text}
    ]

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            endpoint = "https://api.groq.com/openai/v1/chat/completions"
            res = requests.post(
                endpoint,
                json={
                    "model": translation_model,
                    "max_tokens": 6000,
                    "temperature": 0.7,
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )

            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()
    return output

def inject_references_to_messages(
    messages,
    references,
):
    messages = copy.deepcopy(messages)
    system = f"""Nhiệm vụ của bạn là tổng hợp các phản hồi từ nhiều mô hình tham chiếu thành một câu trả lời hoàn chỉnh và chất lượng cao. Hãy đánh giá kỹ lưỡng độ chính xác, mức độ liên quan và các tiềm năng thiên vị trong mỗi phản hồi. Câu trả lời của bạn cần được cấu trúc rõ ràng, mạch lạc, sử dụng các từ nối để tạo sự liền mạch và đảm bảo cung cấp thông tin đầy đủ và chính xác nhất cho người dùng. Hãy viết như cách viết sách hoặc blog và giữ nguyên các thuật ngữ chuyên ngành nếu có.

Responses from models:"""

    for i, reference in enumerate(references):
        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":
        messages[0]["content"] += "\n\n" + system
    else:
        messages = [{"role": "system", "content": system}] + messages

    return messages

def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=6000,
    temperature=0.7,
    generate_fn=generate_together,
):
    if len(references) > 0:
        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
