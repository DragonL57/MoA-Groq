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
    max_tokens=8192,
    temperature=0.6,
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
    max_tokens=8192,
    temperature=0.6,
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
    max_tokens=8192,
    temperature=0.6,
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

def inject_references_to_messages(
    messages,
    references,
):
    messages = copy.deepcopy(messages)
    system = f"""Your task is to synthesize multiple responses into a single, high-quality answer. Critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability. Make sure you stick to the user's language.

Steps to Follow:
Gather Responses: Collect all the responses provided by the models.
Evaluate Information: Critically assess the accuracy, relevance, and potential biases in each response.
Extract Key Points: Identify the most accurate and relevant information from each response.
Refine Content: Combine the extracted points into a coherent and comprehensive answer.
Ensure Clarity and Structure: Make sure the final response is well-structured and easy to understand.
Maintain Accuracy and Reliability: Double-check the synthesized information for any errors or inconsistencies.
Language Consistency: Ensure the response is in the same language as the user's query.

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
    max_tokens=2048,
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
