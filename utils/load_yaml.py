import yaml


def _load_yaml() -> dict:
    with open(r"config/hyperparams.yaml", "r") as file:
        data = yaml.safe_load(file)

    return data
