import json
import logging
import random
from time import sleep

from sklearn.metrics import accuracy_score

from mapper.prompt_mapper import insert_prompt, select_prompt_by_task_id_and_father_id
from utils.llm_util import request_online_llm_api, request_llm_api

optimizing_prompt_system_prompt = """
你是一位提示词优化专家。你的任务是优化提示词，来让模型更好的完成工作，注意：
1.原本提示词中的输出格式不要进行改变
2.直接输出提示词，不要有任何其他内容
3.实际数据仅供给你参考，以便更好的修改提示词，不要让其出现在提示词中
   ---"""

classification_system_prompt_test = """你是一位智能评论分析专家。你的任务是判断当前评论的情感属性.0代表负向，1代表负正。请注意的你的回答要严格按照如下格式进行：
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


def use_prompt_classification(classification_system_prompt, content):
    messages = [{
        "role": "system",
        "content": classification_system_prompt},
        {"role": "user", "content": "评论:\n" + content}]
    result = request_llm_api(messages)
    try:
        result_json = json.loads(result[result.find("{"):result.rfind("}") + 1])
        label = int(result_json.get("label"))
        reasoning = result_json.get("reasoning")
        print(f"label:{label}, reasoning:{reasoning}")
        return label
    except Exception as e:
        logging.error(e)
        return -1


def get_prompt_score(X, y, prompt):
    y_pred = []
    for text in X:
        label = use_prompt_classification(prompt, text)
        y_pred.append(label)
    score = accuracy_score(y, y_pred)
    logging.info(f"计算后分数：{score}")
    return score


def optimizing_prompt(prompt: str, X_train, y_train):
    true_data = ""
    prompt = prompt.replace("\"", "\\\"")
    for i in range(3):
        idx = random.randint(0, len(X_train) - 1)
        x_sample = X_train[idx]
        y_sample = y_train[idx]
        true_data += (f"{(i + 1)}.评论：\n{x_sample}\n结果："
                      "{\"label\": \""
                      f"{y_sample}\n"
                      ", \"reasoning\": \"\"}\n ")

    content = "提示词：\n" + prompt + "\n-------------------------\n" + "实际数据：" + true_data

    messages = [{
        "role": "system",
        "content": optimizing_prompt_system_prompt},
        {"role": "user", "content": content}]
    result = request_online_llm_api(messages)
    # result_json = json.loads(result[result.find("{"):result.rfind("}") + 1])
    # result_prompt = result_json.get("prompt")
    return result


def get_son_prompt(prompt_id, father_prompt, father_id, task_id, X_train, y_train, X_test, y_test):
    son_prompt_list = []
    for i in range(8):
        try:
            prompt = optimizing_prompt(father_prompt, X_train, y_train)
            score = get_prompt_score(X_test, y_test, prompt)
            son_prompt_list.append({
                "prompt": prompt,
                "score": str(score * 1000),
                "father_id": prompt_id,
                "task_id": task_id
            })
            insert_prompt(prompt, score, prompt_id, task_id)
        except Exception as e:
            logging.error(e)
    return son_prompt_list


def iterate_prompt(task_id, father_id, X_train, y_train, X_test, y_test):
    father_prompt_list = select_prompt_by_task_id_and_father_id(task_id, father_id)
    for prompt in father_prompt_list:
        prompt_id = prompt["id"]
        prompt_content = prompt["content"]
        prompt_score = prompt["score"]
        son_prompt_list = get_son_prompt(prompt_id, prompt_content, father_id, task_id, X_train, y_train, X_test,
                                         y_test)
        print(son_prompt_list)
