import yaml
from pathlib import Path


def default_path(filename):
    if filename.startswith('/'):
        return Path(filename)
    else:
        return Path('.') / filename

# load arguments from config.yaml
def load_config(filename='config.yaml'):
    if not Path(default_path(filename)).exists():
        return {}
    with open(default_path(filename), 'r') as f:
        try:
            yaml_f = yaml.load(f, Loader=yaml.FullLoader)
            return yaml_f
        except yaml.YAMLError as exc:
            print(exc)

# save config to config.yaml
def save_config(config, config_to_update={}, filename='config.yaml'):
    assert isinstance(config, dict)
    
    filepath = default_path(filename)
    if filepath.exists():
        if config_to_update is None:
            config_to_update = load_config(filename)
    else:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()

    with filepath.open('w') as f:
        try:
            config_to_update.update(config)
            yaml.dump(config_to_update, f)
        except yaml.YAMLError as exc:
            print(exc)
    