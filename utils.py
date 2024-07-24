import os
import json
import requests
import openai
import copy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt

nltk.download('punkt')
nltk.download('stopwords')

load_dotenv()

DEBUG = int(os.environ.get("DEBUG", "0"))

@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6))
def generate_together(
    model,
    messages,
    max_tokens=4096,
    temperature=0.7,
    streaming=True,
):
    output = None

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
            timeout=10  # Timeout added
        )
        res.raise_for_status()
        if "error" in res.json():
            logger.error(res.json())
            if res.json()["error"]["type"] == "invalid_request_error":
                logger.info("Input + output is longer than max_position_id.")
                return None

        output = res.json()["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout as e:
        logger.error("Timeout error: ", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error: ", e)
        raise
    except Exception as e:
        logger.error("General error: ", e)
        if DEBUG:
            logger.debug(f"Msgs: `{messages}`")
        raise

    if output is None:
        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output

@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6))
def generate_openai(
    model,
    messages,
    max_tokens=4096,
    temperature=0.7,
):
    api_key = os.environ.get('OPENAI_API_KEY')

    if api_key is None:
        logger.error("OPENAI_API_KEY is not set")
        return None

    client = openai.OpenAI(
        api_key=api_key,
    )

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

    except requests.exceptions.Timeout as e:
        logger.error("Timeout error: ", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error: ", e)
        raise
    except Exception as e:
        logger.error("General error: ", e)
        raise

    output = output.strip()

    return output

@retry(wait=wait_exponential(multiplier=1, min=1, max=60), stop=stop_after_attempt(6))
def translate_text(text, translation_model):
    api_key = os.environ.get('GROQ_API_KEY')

    if api_key is None:
        logger.error("GROQ_API_KEY is not set")
        return None

    prompt = {
        "role": "system",
        "content": "Hãy dịch chính xác phản hồi sau đây sang ngôn ngữ của người dùng, giữ nguyên các thuật ngữ chuyên ngành và đảm bảo rằng ý nghĩa và ngữ cảnh ban đầu được giữ nguyên. Đảm bảo rằng bản dịch rõ ràng và dễ hiểu. Trả lời dưới dạng các đoạn văn hoàn chỉnh, chi tiết và cụ thể."
    }

    messages = [
        prompt,
        {"role": "user", "content": text}
    ]

    try:
        endpoint = "https://api.groq.com/openai/v1/chat/completions"
        res = requests.post(
            endpoint,
            json={
                "model": translation_model,
                "max_tokens": 4096,
                "temperature": 0.7,
                "messages": messages,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            timeout=10  # Timeout added
        )
        res.raise_for_status()
        if "error" in res.json():
            logger.error(res.json())
            if res.json()["error"]["type"] == "invalid_request_error":
                logger.info("Input + output is longer than max_position_id.")
                return None

        output = res.json()["choices"][0]["message"]["content"]

    except requests.exceptions.Timeout as e:
        logger.error("Timeout error: ", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error: ", e)
        raise
    except Exception as e:
        logger.error("General error: ", e)
        raise

    output = output.strip()
    return output

def inject_references_to_messages(
    messages,
    references,
):
    messages = copy.deepcopy(messages)
    system = """Bạn đã nhận được nhiều phản hồi từ các mô hình mã nguồn mở khác nhau cho truy vấn mới nhất. Nhiệm vụ của bạn là tổng hợp các phản hồi này thành một câu trả lời duy nhất, chất lượng cao. Hãy đánh giá kỹ lưỡng thông tin, nhận ra rằng một số có thể thiên vị hoặc sai lầm. Đừng sao chép nguyên văn mà hãy cung cấp một câu trả lời tinh chỉnh, chính xác và toàn diện. Đảm bảo câu trả lời của bạn được cấu trúc tốt, mạch lạc, và chính xác. Đối với các công thức toán học hoặc các biểu thức kỹ thuật, hãy đảm bảo rằng chúng được bao quanh bởi ký tự $$ để hiển thị đúng định dạng LaTeX.

Các câu trả lời từ các mô hình:"""

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
    max_tokens=4096,
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

def google_search(query, num_results=10):  # Increase number of search results
    api_key = os.environ.get('GOOGLE_API_KEY')
    cse_id = os.environ.get('GOOGLE_CSE_ID')
    if not api_key or not cse_id:
        raise ValueError("Google API key or Custom Search Engine ID is missing")

    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": num_results
    }

    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        search_results = response.json()
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 400:
            logger.error("Bad Request: ", err)
        elif err.response.status_code == 401:
            logger.error("Unauthorized: ", err)
        elif err.response.status_code == 403:
            logger.error("Forbidden: ", err)
        raise
    except requests.exceptions.Timeout as e:
        logger.error("Timeout error: ", e)
        raise
    except requests.exceptions.RequestException as e:
        logger.error("HTTP error: ", e)
        raise

    return search_results

def extract_snippets(search_results):
    snippets = []
    if "items" in search_results:
        for item in search_results["items"]:
            snippets.append(item["snippet"])
    return snippets

def extract_full_texts(search_results):
    full_texts = []
    if "items" in search_results:
        for item in search_results["items"]:
            full_texts.append(item["snippet"] + "\n\n" + item["link"])
    return full_texts

def extract_keywords(text):
    # Tokenize và loại bỏ stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    keywords = [word for word in word_tokens if word.isalnum() and word not in stop_words]
    return keywords

def expand_query(conversation_history, current_query):
    # Trích xuất từ khóa từ lịch sử cuộc trò chuyện
    history_keywords = extract_keywords(conversation_history)
    
    # Trích xuất từ khóa từ câu hỏi hiện tại
    current_keywords = extract_keywords(current_query)
    
    # Kết hợp và loại bỏ trùng lặp
    all_keywords = list(set(history_keywords + current_keywords))
    
    # Tạo query mở rộng
    expanded_query = " ".join(all_keywords)
    
    return expanded_query
