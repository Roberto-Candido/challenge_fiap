import yaml

def carregar_config_yaml(caminho="config.yaml"):
    with open(caminho, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
