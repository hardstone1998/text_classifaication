from classification_model.embedding_model.service import embedding_svm_sentiment_classification
from utils.llm_util import request_llm_api, request_embedding_api
from utils.logging_config import setup_logging

setup_logging()

if __name__ == '__main__':
    embedding_svm_sentiment_classification()
    # content = "我不关心这个原配小三的问题，我只是坚决不支持这种行为。"
    # result = request_embedding_api(content)
    # print(result)