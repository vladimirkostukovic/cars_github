import pandas as pd
import json
import re
from sqlalchemy import create_engine
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Read config.json
config_path = Path(__file__).resolve().parent / 'config.json'
try:
    with config_path.open(encoding='utf-8') as f:
        cfg = json.load(f)
except Exception:
    with open(str(config_path), 'r') as f:
        cfg = json.load(f)

engine = create_engine(
    f"postgresql+psycopg2://{cfg['USER']}:{cfg['PWD']}@{cfg['HOST']}:{cfg['PORT']}/{cfg['DB']}"
)
SCHEMA = 'acars'

def extract_json_field(val, key):
    if pd.isnull(val):
        return None
    if isinstance(val, dict):
        return val.get(key)
    try:
        return json.loads(val).get(key)
    except Exception:
        return None

def extract_equipment(full_name):
    if not isinstance(full_name, str):
        return None
    parts = [p.strip() for p in full_name.split(',')]
    if len(parts) <= 1:
        return None
    equipment = ', '.join(parts[1:])
    bad = [
        r'\bČR\b', r'\b1\.?maj\b', r'\bmajitel\b', r'koupeno v čr', r'\bAC\b',
        r'\bklima\b', r'\bserv\.?kniha\b', r'\belectric\b', r'\bautomat\b', r'\broadster\b',
        r'\bcabrio\b', r'\bavangarde\b', r'\btourer\b'
    ]
    for b in bad:
        equipment = re.sub(b, '', equipment, flags=re.IGNORECASE)
    equipment = re.sub(r'[ ,]+', ' ', equipment).strip()
    return equipment or None

def extract_motor(name):
    if not isinstance(name, str):
        return None
    pats = [
        r'(\d{1,2}[.,]\d{1,2})\s*(Turbo|TSI|TFSI|CDTI|CDI|i|D)?',
        r'\b(\d{1,2}[.,]\d{1,2})\b',
        r'\b(\d{3})\s?(CDI|d|i)\b',
        r'\b(\d{1,2})\s?V\b',
    ]
    for pat in pats:
        m = re.search(pat, name, re.IGNORECASE)
        if m:
            res = m.group(1).replace(',', '.')
            try:
                if float(res) > 10:
                    continue
            except Exception:
                pass
            return res
    return None

def extract_power(name):
    if not isinstance(name, str):
        return None
    pats = [
        r'([0-9]{2,4})\s*[kK][wW]',
        r'([0-9]{2,4})\s*[hH][pP]',
        r'([0-9]{2,4})\s*[pP][sS]',
        r'([0-9]{2,4})\s*[kK][vV]',
    ]
    for pat in pats:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None

def main():
    log.info('Reading merged table...')
    df_merged = pd.read_sql(f'SELECT * FROM {SCHEMA}.merged', engine)
    log.info(f'Read {len(df_merged)} rows from merged')

    sold_mask = df_merged['date_sold'].notnull()
    sold_count = sold_mask.sum()
    active_count = (~sold_mask).sum()
    log.info(f'In merged: active={active_count}, sold={sold_count}')

    log.info('Extracting motor and power from name...')
    df_merged['motor'] = df_merged['name'].apply(extract_motor)
    df_merged['power'] = df_merged['name'].apply(extract_power)

    standart = pd.DataFrame({
        'id': df_merged['id'],
        'internal_id': df_merged['internal_id'],
        'additional_model_name': df_merged['additional_model_name'],
        'custom_id': df_merged['custom_id'],
        'deal_type': df_merged['deal_type'],
        'manuf_date': df_merged['manufacturing_date'],
        'cebia': df_merged['is_cebia_smart_code_url_verified'],
        'name': df_merged['name'],
        'price': df_merged['price'],
        'tachometr': df_merged['tachometer'],
        'category': df_merged['category_data'].apply(lambda x: extract_json_field(x, 'name')),
        'fuel': df_merged['fuel_data'].apply(lambda x: extract_json_field(x, 'name')),
        'gearbox': df_merged['gearbox_data'].apply(lambda x: extract_json_field(x, 'name')),
        'locality': df_merged['locality_data'].apply(lambda x: extract_json_field(x, 'region')),
        'manufacture': df_merged['manufacturer_data'].apply(lambda x: extract_json_field(x, 'name')),
        'model': df_merged['model_data'].apply(lambda x: extract_json_field(x, 'name')),
        'body': df_merged['category_data'].apply(lambda x: extract_json_field(x, 'seo_name')),
        'equipment': df_merged['name'].apply(extract_equipment),
        'motor': df_merged['motor'],
        'power': df_merged['power'],
        'seller': df_merged['premise_data'].apply(lambda x: extract_json_field(x, 'name')),
        'dated_added': df_merged['date_added'],
        'date_sold': df_merged['date_sold']
    })

    log.info(f'Final dataframe for standart built: {len(standart)} rows, {standart.shape[1]} columns')

    # Diff vs current standart
    try:
        df_old = pd.read_sql(f'SELECT internal_id FROM {SCHEMA}.standart', engine)
        old_ids = set(df_old['internal_id'])
        new_ids = set(standart['internal_id'])
        added = len(new_ids - old_ids)
        dropped = len(old_ids - new_ids)
        log.info(f'Delta: new ads in merged vs previous standart: added={added}, removed={dropped}')
    except Exception:
        log.info('standart not found, creating from scratch.')
        added = len(standart)
        dropped = 0

    standart.to_sql('standart', engine, schema=SCHEMA, if_exists='replace', index=False, method='multi')
    log.info('standart updated.')

    sold_now = standart['date_sold'].notnull().sum()
    active_now = standart['date_sold'].isnull().sum()
    log.info(f'In standart: active={active_now}, sold={sold_now}')

if __name__ == '__main__':
    main()