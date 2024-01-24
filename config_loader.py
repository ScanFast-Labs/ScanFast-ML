import yaml

def load_config(file_path):
    with open(file_path, "r") as config_file:
        return yaml.safe_load(config_file)
