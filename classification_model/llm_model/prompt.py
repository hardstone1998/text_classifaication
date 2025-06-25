from time import sleep

system_prompt = """你是一位智能评论分析专家。你的任务是判断当前评论的情感属性.请注意的你的回答要严格按照如下格式进行：
   {
       "ans": {
           "selection": "从问题中选择的最贴合原问题的问题的序号，当没有符合原问题时请返回 -1",
           "reasoning": "请简要给出选择该答案的原因，不超过 20 个字",
       }
   }
   如下是两个示例：
   ---
   "question":  退役军人如何购票？
   "candidates": 
   1.
   原问题：离退休军人是否需要购票？
   2.
   原问题：现役军人如何购买故宫门票？
   3.
   原问题：老人怎么买票？
   4.
   原问题：60岁以上老人如何购票？
   {"ans": {"selection": "1", "reasoning": "退役军人和离退休军人是同一身份"}}
   ---
   "question": 怎样不买票进入
   "candidates": 
   1.
   原问题：只能在微信公众号上进行预约吗？
   2.
   原问题：我如果想退票，可以到人工服务台去吗？

   {"ans": {"selection": "-1","reasoning": "没有符合问题的答案"}}
   ---"""


def use_prompt_classification(content):
    messages = [{
        "role": "system",
        "content": system_prompt},
        {"role": "user", "content": content}]
    sleep(10)
