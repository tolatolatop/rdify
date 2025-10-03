import logging.config
from dotenv import load_dotenv
from pathlib import Path
from ruamel.yaml import YAML

PACKAGE_ROOT = Path(__file__).parent

load_dotenv()

yaml = YAML()

with open(PACKAGE_ROOT / "log_config.yaml", "r") as f:
    config = yaml.load(f)

logs_dir = Path('.') / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)

logging.config.dictConfig(config)

config = {}

if __name__ == "__main__":
    logger = logging.getLogger("app")
    logger.info("Hello, world!")
