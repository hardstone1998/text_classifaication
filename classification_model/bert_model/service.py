import logging
import torch

from sklearn.metrics import accuracy_score

from classification_model.bert_model.train_bert import load_bert_model, train_and_save_bert_model, encode_bert_tensor


def bert_C_MTEB_OnlineShopping_classification():
    # Test Accuracy: 0.9400
    dataset_name = "C-MTEB/OnlineShopping-classification"
    model_name = "bert_OnlineShopping_classification"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    train_loader, test_loader = encode_bert_tensor(dataset_name)

    model = load_bert_model(model_name, device)

    if model is None:
        model = train_and_save_bert_model(train_loader, model_name, device)

    # ✅ 评估
    model.eval()
    all_preds, all_labels = [], []


    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"])

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")