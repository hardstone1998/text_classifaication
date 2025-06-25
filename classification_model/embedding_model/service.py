import logging

import torch

from classification_model.embedding_model.MLPClassifier import train_and_save_mlp_model, load_mlp_model
from classification_model.embedding_model.svm import train_and_save_model, load_model
from handle_data.download_data import Liuyu_sentiment_classification, C_MTEB_OnlineShopping_classification
from sklearn.metrics import accuracy_score

from utils.llm_util import request_embedding_api


def embedding_svm_sentiment_classification():
    # 数据质量较差 准确率Accuracy: 0.7993333333333333
    # X_train_text, y_train, X_test_text, y_test = Liuyu_sentiment_classification()
    # model_name = "svm_sentiment_classification"

    #  准确率Accuracy: 0.949
    X_train_text, y_train, X_test_text, y_test = C_MTEB_OnlineShopping_classification()
    model_name = "svm_OnlineShopping_classification"
    clf = load_model(model_name)
    if clf is None:
        X_train = []
        for content in X_train_text:
            X_embedding = request_embedding_api(content)
            X_train.append(X_embedding)
        clf = train_and_save_model(X_train, y_train, model_name)
    X_test = []
    for content in X_test_text:
        X_embedding = request_embedding_api(content)
        X_test.append(X_embedding)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


def embedding_mlp_sentiment_classification():
    # X_train_text, y_train, X_test_text, y_test = Liuyu_sentiment_classification()
    # model_name = "mlp_sentiment_classification"

    #  准确率Accuracy: 0.933
    X_train_text, y_train, X_test_text, y_test = C_MTEB_OnlineShopping_classification()
    model_name = "svm_OnlineShopping_classification"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 获取嵌入
    X_train = [request_embedding_api(text) for text in X_train_text]
    X_test = [request_embedding_api(text) for text in X_test_text]

    input_dim = len(X_train[0])
    model = load_mlp_model(model_name, input_dim, device)

    if model is None:
        model = train_and_save_mlp_model(X_train, y_train, model_name, device)

    # 推理
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_test_tensor)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    print("Accuracy:", accuracy_score(y_test, preds))
