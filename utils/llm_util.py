import time
from utils.load_config import load_config
from openai import OpenAI
import logging

config = load_config()
openai_conf = config.get('openai')
llm_conf = openai_conf.get('llm')
llm_url = llm_conf.get('url')
llm_model = llm_conf.get('model')
embedding_conf = openai_conf.get('embedding')
embedding_url = embedding_conf.get('url')
embedding_model = embedding_conf.get('model')

client_llm = OpenAI(
    api_key="EMPTY",  # 替换成真实DashScope的API_KEY
    base_url=llm_url,  # 填写DashScope服务endpoint
)
client_embedding = OpenAI(
    api_key="EMPTY",  # 替换成真实DashScope的API_KEY
    base_url=embedding_url,  # 填写DashScope服务endpoint
)


def request_llm_api(content):
    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    t1 = time.time()
    completion = client_llm.chat.completions.create(model=llm_model, messages=messages)
    t2 = time.time()
    result = completion.choices[0].message.content
    logging.info(f"大模型请求时间：{t2 - t1}")
    logging.info(f"大模型返回内容：{result}")
    return result


def request_embedding_api(content):
    messages = [content]
    t1 = time.time()
    completion = client_embedding.embeddings.create(model=embedding_model, input=messages)
    result = completion.data[0].embedding
    t2 = time.time()
    logging.info(f"embedding请求时间：{t2 - t1}")
    return result
