from classification_model.embedding_model.service import embedding_svm_sentiment_classification, \
    embedding_mlp_sentiment_classification
from utils.llm_util import request_llm_api, request_embedding_api
from utils.logging_config import setup_logging

setup_logging()

if __name__ == '__main__':
    # embedding_svm_sentiment_classification()
    embedding_mlp_sentiment_classification()