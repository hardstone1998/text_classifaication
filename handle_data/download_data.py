from datasets import load_dataset

def Liuyu_sentiment_classification():
    dataset_name = "Liuyu/sentiment-classification"
    dataset = load_dataset(dataset_name)
    X_train = dataset['train']['text']
    y_train = dataset['train']['label']
    X_test = dataset['test']['text']
    y_test = dataset['test']['label']
    return X_train, y_train, X_test, y_test