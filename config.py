"""
Конфигурация и утилиты.
"""
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import pandas as pd
from sqlalchemy import create_engine, text


# ============== Константы ==============
SCHEMA = 'acars'
MIN_PRICE = 30000
MAX_PRICE = 5_000_000
MIN_YEAR = 2015
MAX_YEAR = 2024
RECO_DISCOUNT = 0.9
RECO_FLAT = 13000
MIN_SALES = 5
LIQ_DAYS = 30
MIN_PRICE_CHANGE = 1000

SNAPSHOT_PATTERN = 'sauto_%'
SNAPSHOT_REGEX = r'sauto_(\d{2})(\d{2})(\d{4})'

GROUP_COLS = ['manufacture', 'model', 'rok_vyroby', 'fuel', 'gearbox', 'skupina_tachometru']

JSON_COLUMNS = [
    'category_data', 'fuel_data', 'gearbox_data', 'locality_data',
    'manufacturer_data', 'model_data', 'premise_data', 'user_data'
]


# ============== Логирование ==============
def setup_logging(log_file: str = 'pipeline.log', level=logging.INFO):
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    console_fmt = '[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(console_fmt, datefmt=date_fmt))

    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(console_fmt, datefmt=date_fmt))

    root.setLevel(level)
    root.addHandler(ch)
    root.addHandler(fh)

    logger = logging.getLogger('Setup')
    logger.info('=' * 60)
    logger.info(f'Pipeline started: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logger.info('=' * 60)


# ============== База данных ==============
def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent / 'config.json'
    with config_path.open(encoding='utf-8') as f:
        return json.load(f)


def create_db_engine(config: dict):
    conn_str = (
        f"postgresql+psycopg2://{config['USER']}:{config['PWD']}"
        f"@{config['HOST']}:{config['PORT']}/{config['DB']}"
    )
    return create_engine(conn_str)


def get_engine():
    """Создать engine из config.json."""
    config = load_config()
    return create_db_engine(config)