import mysql.connector
from dbutils.pooled_db import PooledDB
import logging
from utils.load_config import load_config


class ConnectMysqlPool:
    """
    连接MySQL数据库的连接池类。

    属性:
    db_account (dict): 数据库账号信息，包括用户名和密码等。
    db (str): 数据库名称。
    pool (PooledDB): MySQL连接池对象。

    方法:
    __init__: 初始化连接池类实例。
    _obtaining_data: 从配置文件中获取测试数据。
    create_mysql_pool: 创建MySQL连接池。
    get_conn: 从连接池中获取一个连接。
    close: 关闭数据库连接和游标。
    execute: 使用连接执行SQL语句。
    """

    def __init__(self):
        """
        初始化连接池类实例。

        参数:
        db (str): 测试库名称。
        db_account (dict): 包含数据库账号信息的字典。
        """
        config = load_config()
        mysql = config.get('mysql')
        url = mysql.get("url")
        port = mysql.get("port")
        username = mysql.get("username")
        password = mysql.get("password")
        db_name = mysql.get('database')

        self.db_account = {
            'host': url,
            'port': int(port),
            'user': username,
            'password': password,
            "charset": "utf8",
            "auth_plugin": 'mysql_native_password'
        }
        self.db = db_name  # 测试库

        # 创建连接池
        self.pool = self.create_mysql_pool()

    # 创建MySQL连接池
    def create_mysql_pool(self):
        """
        根据配置信息创建MySQL连接池。

        返回:
        PooledDB: MySQL连接池对象。
        """
        pool = PooledDB(
            creator=mysql.connector,
            **self.db_account,
            database=self.db
        )
        return pool

    # 从连接池中获取一个连接
    def get_conn(self):
        """
        从连接池中获取一个数据库连接。

        返回:
        connection: 数据库连接对象。
        """
        return self.pool.connection()

    # 关闭数据库连接和游标
    def close(self, conn, cursor):
        """
        关闭数据库连接和游标。

        参数:
        conn (connection): 数据库连接对象。
        cursor (cursor): 数据库游标对象。
        """
        try:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        except Exception as e:
            logging.error(f"关闭连接或游标时发生错误：{e}")

    # 使用连接执行sql
    def execute(self, sql, params: dict = None):
        """
        使用获取的连接执行SQL语句。

        参数:
        sql (str): SQL语句。
        params (tuple): SQL参数。

        返回:
        list: 执行SQL语句后的结果集，若执行出错则返回None。
        """
        conn = self.get_conn()
        cursor = conn.cursor(dictionary=True)  # 设置为字典形式返回

        try:
            if params is not None:
                logging.info(f"参数:{params}")
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            if cursor.with_rows:  # 检查是否有结果集需要读取
                result = cursor.fetchall()
                conn.commit()
                return result
            else:
                conn.commit()
                return []  # 如果没有结果集，返回空列表
        except Exception as e:
            logging.error(f"表：{self.db}，执行sql：{sql}，报错：{e}")
            conn.rollback()
            raise
        finally:
            self.close(conn, cursor)
