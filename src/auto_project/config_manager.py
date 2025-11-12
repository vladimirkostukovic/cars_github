
import json
import logging
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine
from typing import Optional


class ConfigManager:

    
    def __init__(self, config_path: Optional[Path] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if config_path is None:
            config_path = Path(__file__).resolve().parent / 'config.json'
        
        self.config_path = config_path
        self.config = self._load_config()
        self.schema = 'acars'
        
    def _load_config(self) -> dict:
        try:
            with self.config_path.open(encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            with open(str(self.config_path), 'r') as f:
                return json.load(f)
    
    def get_engine(self):
        conn_str = (
            f"postgresql+psycopg2://{self.config['USER']}:{self.config['PWD']}"
            f"@{self.config['HOST']}:{self.config['PORT']}/{self.config['DB']}"
        )
        return create_engine(conn_str)
    
    def get(self, key: str, default=None):
        return self.config.get(key, default)


def setup_logging(log_file: str = 'pipeline.log', level=logging.INFO):
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    
    # Color formatter for console
    class ColoredFormatter(logging.Formatter):
        COLORS = {
            'DEBUG': '\033[36m', 'INFO': '\033[32m', 'WARNING': '\033[33m',
            'ERROR': '\033[31m', 'CRITICAL': '\033[35m'
        }
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        def format(self, record):
            color = self.COLORS.get(record.levelname, self.RESET)
            record.levelname_colored = f"{color}{record.levelname:8s}{self.RESET}"
            record.name_colored = f"{self.BOLD}{record.name}{self.RESET}"
            return super().format(record)
    
    console_fmt = '[%(asctime)s] [%(levelname_colored)s] [%(name_colored)s] %(message)s'
    file_fmt = '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(ColoredFormatter(console_fmt, datefmt=date_fmt))
    
    # File handler
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(file_fmt, datefmt=date_fmt))
    
    # Setup root logger
    root.setLevel(level)
    root.addHandler(ch)
    root.addHandler(fh)
    
    # Log startup
    logger = logging.getLogger('Setup')
    logger.info('='*80)
    logger.info(f'Logging initialized - File: {log_file}')
    logger.info(f'Session: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('='*80)
