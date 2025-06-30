from utils.ConnectMysqlPool import ConnectMysqlPool
from utils.load_config import load_config
import logging

cmp = ConnectMysqlPool()


def select_prompt_by_task_id_and_father_id(task_id: str, father_id: str):
    try:
        sql = "select * from prompt where task_id = %s and father_id = %s"
        params = (task_id, father_id)
        rs = cmp.execute(sql, params)
        logging.info(f"查询数据库成功,result:{rs}")
    except Exception as e:
        logging.error("查询数据库异常")
        raise Exception(e)
    return rs


# 逐条插入论文
def insert_prompt(content, score, father_id, task_id):
    try:
        sql = """
        INSERT INTO prompt (content, score, father_id, task_id,create_time)
        VALUES (%s, %s, %s, %s, now())
        """

        rs = cmp.execute(sql, (
            content, int(score * 1000), father_id, task_id))
        logging.info(f"插入数据库成功,result:{rs}")
    except Exception as e:
        logging.error(f"Error : {str(e)}")
        print(e)
        raise  # 可选择是否终止或跳过继续
