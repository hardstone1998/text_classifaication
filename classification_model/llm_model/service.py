from sklearn.metrics import accuracy_score

from classification_model.llm_model.prompt import use_prompt_classification, get_prompt_score, iterate_prompt
from handle_data.download_data import C_MTEB_OnlineShopping_classification
from mapper.prompt_mapper import insert_prompt


def llm_prompt_sentiment_classification(father_id, task_id):
    #  准确率Accuracy: 0.931
    X_train, y_train, X_test, y_test = C_MTEB_OnlineShopping_classification()
    classification_system_prompt = """你是一位智能评论分析专家。你的任务是判断当前评论的情感属性.0代表负向，1代表负正。请注意的你的回答要严格按照如下格式进行：
       {
           "label": 0或1,
           "reasoning": "请简要给出选择该答案的原因，不超过 20 个字",
       }
       如下是两个示例：
       ---
       评论：
       裤子才穿了几天 一张腿 直接破了 什么质量啊 而且 掉色特别严重 手一摸 满手都是黑？
       回答：
       {"label": 0, "reasoning": "对裤子质量有严重批评"}
       ---
       评论：
       质量还行，这个价位还是不错的
       回答：
       {"label": 1, "reasoning": "客户认为裤子性价比高"}
       ---"""
    score = get_prompt_score(X_test, y_test, classification_system_prompt)
    insert_prompt(classification_system_prompt, str(score), 0, 1)
    iterate_prompt(task_id, father_id, X_train, y_train, X_test, y_test)


def llm_iterate_prompt_sentiment_classification(father_id, task_id):
    #  准确率Accuracy: 0.931
    X_train, y_train, X_test, y_test = C_MTEB_OnlineShopping_classification()
    iterate_prompt(task_id, father_id, X_train, y_train, X_test, y_test)
