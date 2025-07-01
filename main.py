from classification_model.embedding_model.service import embedding_svm_C_MTEB_OnlineShopping_classification, \
    embedding_mlp_C_MTEB_OnlineShopping_classification
from classification_model.llm_model.service import llm_prompt_sentiment_classification, \
    llm_iterate_prompt_sentiment_classification
from handle_data.download_data import C_MTEB_OnlineShopping_classification
from mapper.prompt_mapper import insert_prompt
from utils.llm_util import request_llm_api, request_embedding_api
from utils.logging_config import setup_logging

setup_logging()

if __name__ == '__main__':
    # embedding_svm_C_MTEB_OnlineShopping_classification()
    # embedding_mlp_C_MTEB_OnlineShopping_classification()

    # X_train, y_train, X_test, y_test = C_MTEB_OnlineShopping_classification()
    # result = optimizing_prompt(classification_system_prompt, X_train, y_train)
    # print(result)
    # llm_prompt_sentiment_classification()

    # llm_prompt_sentiment_classification()
    llm_iterate_prompt_sentiment_classification(0, 1)