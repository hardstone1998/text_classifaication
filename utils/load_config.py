import yaml
def load_config():
    file_path = './conf/config.yml'
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"加载配置文件时出错: {str(e)}")
        # raise Exception("加载配置文件时出错")