import os
import joblib
from sklearn.svm import SVC

from utils.load_config import load_config

config = load_config()
save_models = config.get('save_models')
embedding_dir = "embeddings"


def train_and_save_model(X_train, y_train, model_name: str):
    print("Training model...")
    # 模型保存路径
    save_embedding_full_dir = os.path.join(save_models, "embedding_dir")
    os.makedirs(save_embedding_full_dir, exist_ok=True)

    model_path = os.path.join(save_embedding_full_dir, f"{model_name}.joblib")
    print(f"Saving model to {model_path}")
    # 训练 SVM 模型
    clf = SVC()
    clf.fit(X_train, y_train)

    # 保存模型
    joblib.dump(clf, model_path)
    print(f"模型已保存到: {model_path}")

    return clf


def load_model(model_name: str):
    save_embedding_full_dir = os.path.join(save_models, "embedding_dir")
    os.makedirs(save_embedding_full_dir, exist_ok=True)
    model_path = os.path.join(save_embedding_full_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        return None

    clf = joblib.load(model_path)
    print(f"模型已从 {model_path} 加载成功")
    return clf
