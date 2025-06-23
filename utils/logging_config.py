import logging

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;5;240m"  # DEBUG 级别灰色
    light_white = "\x1b[38;5;250m"  # INFO 级别稍亮一点的白色
    yellow = "\x1b[33;20m"  # WARNING 级别黄色
    red = "\x1b[31;20m"  # ERROR 级别红色
    bold_red = "\x1b[31;1m"  # CRITICAL 级别粗体红色
    reset = "\x1b[0m"  # 重置颜色
    format = "%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: light_white + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def __init__(self):
        super().__init__(fmt=None, datefmt=None, style='%')

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
# 配置基本日志
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 默认设置为 DEBUG 级别
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
