import yaml
from pathlib import Path

def load_config(config_path="configs/baseline.yaml"):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)