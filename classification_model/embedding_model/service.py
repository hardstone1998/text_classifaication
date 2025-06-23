from classification_model.embedding_model.svm import train_and_save_model, load_model
from handle_data.download_data import Liuyu_sentiment_classification
from sklearn.metrics import accuracy_score

from utils.llm_util import request_embedding_api

# 准确率Accuracy: 0.7993333333333333
def embedding_svm_sentiment_classification():
    X_train_text, y_train, X_test_text, y_test = Liuyu_sentiment_classification()
    model_name = "svm_sentiment_classification"
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

# def embedding_mlp_sentiment_classification():
#     X_train_text, y_train, X_test_text, y_test = Liuyu_sentiment_classification()
#     model_name = "mlp_sentiment_pytorch"
#     save_path = os.path.join(save_models, "embedding_dir", f"{model_name}.pt")
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     # 获取嵌入
#     X_train = [request_embedding_api(text) for text in X_train_text]
#     X_test = [request_embedding_api(text) for text in X_test_text]
#
#     input_dim = len(X_train[0])
#     model = MLPClassifier(input_dim).to(device)
#
#     if os.path.exists(save_path):
#         model.load_state_dict(torch.load(save_path, map_location=device))
#         print(f"模型已加载：{save_path}")
#     else:
#         model = train_and_save_pytorch_model(X_train, y_train, model_name, device)
#
#     # 推理
#     model.eval()
#     with torch.no_grad():
#         X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#         logits = model(X_test_tensor)
#         preds = torch.argmax(logits, dim=1).cpu().numpy()
#
#     print("Accuracy:", accuracy_score(y_test, preds))
