#from merged table to standart
#price change tracker

import json
import re
import logging
from datetime import datetime
from typing import Optional, List
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


# ============== Константы ==============
SCHEMA = 'acars'
MIN_PRICE_CHANGE = 1000
SNAPSHOT_PATTERN = 'sauto_%'
SNAPSHOT_REGEX = r'sauto_(\d{2})(\d{2})(\d{4})'


# ============== Инициализация ==============
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def get_engine():
    config_path = Path(__file__).resolve().parent / 'config.json'
    with config_path.open(encoding='utf-8') as f:
        config = json.load(f)
    conn_str = (
        f"postgresql+psycopg2://{config['USER']}:{config['PWD']}"
        f"@{config['HOST']}:{config['PORT']}/{config['DB']}"
    )
    return create_engine(conn_str)


# ============== Утилиты ==============
def extract_json_field(val, key: str) -> Optional[str]:
    if pd.isnull(val):
        return None
    if isinstance(val, dict):
        return val.get(key)
    try:
        return json.loads(val).get(key)
    except Exception:
        return None


def extract_motor(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    patterns = [
        r'(\d{1,2}[.,]\d{1,2})\s*(Turbo|TSI|TFSI|CDTI|CDI|i|D)?',
        r'\b(\d{1,2}[.,]\d{1,2})\b',
        r'\b(\d{3})\s?(CDI|d|i)\b',
        r'\b(\d{1,2})\s?V\b',
    ]
    for pat in patterns:
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


def extract_power(name: str) -> Optional[int]:
    if not isinstance(name, str):
        return None
    patterns = [
        r'([0-9]{2,4})\s*[kK][wW]',
        r'([0-9]{2,4})\s*[hH][pP]',
        r'([0-9]{2,4})\s*[pP][sS]',
        r'([0-9]{2,4})\s*[kK][vV]',
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if m:
            return int(m.group(1))
    return None


def extract_equipment(name: str) -> Optional[str]:
    if not isinstance(name, str):
        return None
    parts = [p.strip() for p in name.split(',')]
    if len(parts) <= 1:
        return None
    equipment = ', '.join(parts[1:])
    bad = [
        r'\bČR\b', r'\b1\.?maj\b', r'\bmajitel\b', r'koupeno v čr',
        r'\bAC\b', r'\bklima\b', r'\bserv\.?kniha\b', r'\belectric\b',
        r'\bautomat\b', r'\broadster\b', r'\bcabrio\b', r'\bavangarde\b', r'\btourer\b'
    ]
    for b in bad:
        equipment = re.sub(b, '', equipment, flags=re.IGNORECASE)
    equipment = re.sub(r'[ ,]+', ' ', equipment).strip()
    return equipment or None


def list_snapshots(engine) -> List[str]:
    sql = text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = :s AND table_name LIKE :p"
    )
    with engine.connect() as conn:
        rows = conn.execute(sql, {'s': SCHEMA, 'p': SNAPSHOT_PATTERN}).fetchall()

    snapshots = []
    for row in rows:
        name = row[0]
        m = re.search(SNAPSHOT_REGEX, name)
        if m:
            date = datetime(int(m[3]), int(m[2]), int(m[1]))
            snapshots.append((name, date))
    snapshots.sort(key=lambda x: x[1])
    return [s[0] for s in snapshots]


def get_snapshot_date(name: str) -> datetime:
    return datetime.strptime(name[-8:], '%d%m%Y')


# ============== Шаг 2: Трансформация ==============
def load_merged_data(engine) -> pd.DataFrame:
    logger = logging.getLogger('transform')
    logger.info('Reading merged table...')
    df = pd.read_sql(f'SELECT * FROM {SCHEMA}.merged', engine)
    logger.info(f'Loaded {len(df)} rows from merged')
    sold = df['date_sold'].notnull().sum()
    active = df['date_sold'].isnull().sum()
    logger.info(f'In merged: active={active}, sold={sold}')
    return df


def transform_to_standart(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('transform')
    logger.info('Extracting motor and power...')

    standart = pd.DataFrame({
        'id': df['id'],
        'internal_id': df['internal_id'],
        'additional_model_name': df.get('additional_model_name'),
        'custom_id': df.get('custom_id'),
        'deal_type': df.get('deal_type'),
        'manuf_date': df.get('manufacturing_date'),
        'cebia': df.get('is_cebia_smart_code_url_verified'),
        'name': df['name'],
        'price': df['price'],
        'tachometr': df.get('tachometer'),
        'category': df['category_data'].apply(lambda x: extract_json_field(x, 'name')),
        'fuel': df['fuel_data'].apply(lambda x: extract_json_field(x, 'name')),
        'gearbox': df['gearbox_data'].apply(lambda x: extract_json_field(x, 'name')),
        'locality': df['locality_data'].apply(lambda x: extract_json_field(x, 'region')),
        'manufacture': df['manufacturer_data'].apply(lambda x: extract_json_field(x, 'name')),
        'model': df['model_data'].apply(lambda x: extract_json_field(x, 'name')),
        'body': df['category_data'].apply(lambda x: extract_json_field(x, 'seo_name')),
        'equipment': df['name'].apply(extract_equipment),
        'motor': df['name'].apply(extract_motor),
        'power': df['name'].apply(extract_power),
        'seller': df['premise_data'].apply(lambda x: extract_json_field(x, 'name')),
        'dated_added': df['date_added'],
        'date_sold': df['date_sold']
    })

    logger.info(f'Standart dataframe: {len(standart)} rows, {standart.shape[1]} cols')
    return standart


def save_standart(engine, df: pd.DataFrame):
    logger = logging.getLogger('transform')
    try:
        df_old = pd.read_sql(f'SELECT internal_id FROM {SCHEMA}.standart', engine)
        old = set(df_old['internal_id'])
        new = set(df['internal_id'])
        logger.info(f'Delta: added={len(new - old)}, removed={len(old - new)}')
    except Exception:
        logger.info('Standart not found, creating from scratch')

    df.to_sql('standart', engine, schema=SCHEMA,
              if_exists='replace', index=False, method='multi')
    logger.info('Standart table updated')

    sold = df['date_sold'].notnull().sum()
    active = df['date_sold'].isnull().sum()
    logger.info(f'In standart: active={active}, sold={sold}')


def run_transform(engine):
    logger = logging.getLogger('transform')
    df_merged = load_merged_data(engine)
    df_standart = transform_to_standart(df_merged)
    save_standart(engine, df_standart)
    logger.info("✓ Transform complete")


# ============== Шаг 3: История цен ==============
def ensure_price_history_table(engine):
    logger = logging.getLogger('prices')
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.price_history (
        id BIGINT,
        old_price NUMERIC,
        new_price NUMERIC,
        change_date DATE
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Price history table checked/created")


def load_snapshot_prices(engine, name: str) -> pd.DataFrame:
    return pd.read_sql(text(f'SELECT id, price FROM {SCHEMA}."{name}"'), engine)


def find_price_changes(df_prev: pd.DataFrame, df_last: pd.DataFrame) -> pd.DataFrame:
    merged = df_last.merge(df_prev, on='id', how='inner', suffixes=('_new', '_old'))
    changes = merged[
        (merged['price_new'] != merged['price_old']) &
        (abs(merged['price_new'] - merged['price_old']) >= MIN_PRICE_CHANGE)
        ].copy()
    return changes


def filter_existing_changes(engine, changes: pd.DataFrame, date) -> pd.DataFrame:
    logger = logging.getLogger('prices')
    try:
        today_hist = pd.read_sql(
            f"SELECT id, change_date FROM {SCHEMA}.price_history WHERE change_date = %(dt)s",
            engine, params={"dt": date}
        )
        mask = ~changes.set_index(['id', 'change_date']).index.isin(
            today_hist.set_index(['id', 'change_date']).index
        )
        changes = changes[mask]
    except Exception as e:
        logger.warning(f"Could not load history for dedup: {e}")
    return changes


def save_price_changes(engine, changes: pd.DataFrame):
    logger = logging.getLogger('prices')
    if changes.empty:
        logger.info("All changes already recorded")
        return
    changes.to_sql('price_history', engine, schema=SCHEMA,
                   if_exists='append', index=False, method='multi', chunksize=5000)
    logger.info(f"Saved {len(changes)} price changes")


def run_price_tracking(engine):
    logger = logging.getLogger('prices')
    snapshots = list_snapshots(engine)

    if len(snapshots) < 2:
        logger.warning("Not enough snapshots to compare")
        return

    ensure_price_history_table(engine)

    prev, last = snapshots[-2], snapshots[-1]
    logger.info(f"Comparing {prev} and {last}")

    df_prev = load_snapshot_prices(engine, prev)
    df_last = load_snapshot_prices(engine, last)
    date_last = get_snapshot_date(last).date()

    changes = find_price_changes(df_prev, df_last)
    if changes.empty:
        logger.info("No price changes found")
        return

    changes['change_date'] = date_last
    changes = changes[['id', 'price_old', 'price_new', 'change_date']]
    changes = changes.rename(columns={'price_old': 'old_price', 'price_new': 'new_price'})

    changes = filter_existing_changes(engine, changes, date_last)
    save_price_changes(engine, changes)
    logger.info("✓ Price tracking complete")


# ============== Sanity Checks ==============
def sanity_check_standart(engine) -> bool:
    logger = logging.getLogger('transform')
    logger.info("Running sanity checks for standart...")

    errors = []
    warnings = []

    # 1. Таблица существует и не пустая
    try:
        count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {SCHEMA}.standart", engine).iloc[0]['cnt']
        if count == 0:
            errors.append("standart table is empty")
        else:
            logger.info(f"  Total records: {count}")
    except Exception as e:
        errors.append(f"Cannot read standart table: {e}")
        return False

    # 2. Проверка на дубликаты id
    dupes = pd.read_sql(f"""
        SELECT id, COUNT(*) as cnt 
        FROM {SCHEMA}.standart 
        GROUP BY id 
        HAVING COUNT(*) > 1
        LIMIT 10
    """, engine)
    if not dupes.empty:
        errors.append(f"Duplicate IDs found: {len(dupes)}")
    else:
        logger.info("  No duplicate IDs")

    # 3. NULL в критичных полях
    critical_cols = ['id', 'price', 'dated_added', 'manufacture', 'model']
    for col in critical_cols:
        try:
            null_count = pd.read_sql(f"""
                SELECT COUNT(*) as cnt 
                FROM {SCHEMA}.standart 
                WHERE {col} IS NULL
            """, engine).iloc[0]['cnt']
            if null_count > 0:
                if col in ['manufacture', 'model']:
                    warnings.append(f"NULL values in {col}: {null_count}")
                else:
                    errors.append(f"NULL values in {col}: {null_count}")
            else:
                logger.info(f"  No NULLs in {col}")
        except Exception:
            pass

    # 4. Проверка price range
    price_stats = pd.read_sql(f"""
        SELECT 
            MIN(CAST(price AS NUMERIC)) as min_price,
            MAX(CAST(price AS NUMERIC)) as max_price,
            AVG(CAST(price AS NUMERIC)) as avg_price
        FROM {SCHEMA}.standart
        WHERE price IS NOT NULL
    """, engine).iloc[0]
    logger.info(f"  Price range: {price_stats['min_price']:.0f} - {price_stats['max_price']:.0f} (avg: {price_stats['avg_price']:.0f})")

    if price_stats['min_price'] < 0:
        errors.append("Negative prices found")
    if price_stats['max_price'] > 50000000:
        warnings.append(f"Very high max price: {price_stats['max_price']}")

    # 5. Active vs Sold
    stats = pd.read_sql(f"""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN date_sold IS NULL THEN 1 ELSE 0 END) as active,
            SUM(CASE WHEN date_sold IS NOT NULL THEN 1 ELSE 0 END) as sold
        FROM {SCHEMA}.standart
    """, engine).iloc[0]
    logger.info(f"  Active: {stats['active']}, Sold: {stats['sold']}")

    # 6. Fuel types check
    fuels = pd.read_sql(f"""
        SELECT fuel, COUNT(*) as cnt 
        FROM {SCHEMA}.standart 
        WHERE fuel IS NOT NULL
        GROUP BY fuel
        ORDER BY cnt DESC
    """, engine)
    logger.info(f"  Fuel types: {len(fuels)}")
    if len(fuels) > 20:
        warnings.append(f"Too many fuel types: {len(fuels)}")

    # 7. Gearbox types check
    gearbox = pd.read_sql(f"""
        SELECT gearbox, COUNT(*) as cnt 
        FROM {SCHEMA}.standart 
        WHERE gearbox IS NOT NULL
        GROUP BY gearbox
    """, engine)
    logger.info(f"  Gearbox types: {len(gearbox)}")

    # Результаты
    if errors:
        logger.error("Sanity check FAILED:")
        for e in errors:
            logger.error(f"  ERROR: {e}")
        for w in warnings:
            logger.warning(f"  WARNING: {w}")
        return False

    if warnings:
        logger.warning("Sanity check PASSED with warnings:")
        for w in warnings:
            logger.warning(f"  WARNING: {w}")
    else:
        logger.info("Sanity check PASSED")

    return True


def sanity_check_price_history(engine) -> bool:
    logger = logging.getLogger('prices')
    logger.info("Running sanity checks for price_history...")

    errors = []
    warnings = []

    # 1. Таблица существует
    try:
        count = pd.read_sql(f"SELECT COUNT(*) as cnt FROM {SCHEMA}.price_history", engine).iloc[0]['cnt']
        logger.info(f"  Total price changes: {count}")
        if count == 0:
            logger.info("  price_history is empty (may be expected)")
            return True
    except Exception as e:
        errors.append(f"Cannot read price_history table: {e}")
        return False

    # 2. Проверка на отрицательные цены
    neg_prices = pd.read_sql(f"""
        SELECT COUNT(*) as cnt 
        FROM {SCHEMA}.price_history 
        WHERE old_price < 0 OR new_price < 0
    """, engine).iloc[0]['cnt']
    if neg_prices > 0:
        errors.append(f"Negative prices in history: {neg_prices}")
    else:
        logger.info("  No negative prices")

    # 3. Проверка что old_price != new_price
    same_price = pd.read_sql(f"""
        SELECT COUNT(*) as cnt 
        FROM {SCHEMA}.price_history 
        WHERE old_price = new_price
    """, engine).iloc[0]['cnt']
    if same_price > 0:
        warnings.append(f"Records with same old/new price: {same_price}")
    else:
        logger.info("  All records have price change")

    # 4. Проверка min change threshold
    small_changes = pd.read_sql(f"""
        SELECT COUNT(*) as cnt 
        FROM {SCHEMA}.price_history 
        WHERE ABS(old_price - new_price) < {MIN_PRICE_CHANGE}
    """, engine).iloc[0]['cnt']
    if small_changes > 0:
        warnings.append(f"Changes below threshold ({MIN_PRICE_CHANGE}): {small_changes}")
    else:
        logger.info(f"  All changes >= {MIN_PRICE_CHANGE}")

    # 5. Дата распределения
    date_range = pd.read_sql(f"""
        SELECT 
            MIN(change_date) as min_date,
            MAX(change_date) as max_date,
            COUNT(DISTINCT change_date) as days
        FROM {SCHEMA}.price_history
    """, engine).iloc[0]
    logger.info(f"  Date range: {date_range['min_date']} to {date_range['max_date']} ({date_range['days']} days)")

    # 6. Средний размер изменения
    avg_change = pd.read_sql(f"""
        SELECT 
            AVG(ABS(old_price - new_price)) as avg_abs,
            AVG((old_price - new_price) / NULLIF(old_price, 0) * 100) as avg_pct
        FROM {SCHEMA}.price_history
    """, engine).iloc[0]
    logger.info(f"  Avg change: {avg_change['avg_abs']:.0f} CZK ({avg_change['avg_pct']:.1f}%)")

    # Результаты
    if errors:
        logger.error("Sanity check FAILED:")
        for e in errors:
            logger.error(f"  ERROR: {e}")
        for w in warnings:
            logger.warning(f"  WARNING: {w}")
        return False

    if warnings:
        logger.warning("Sanity check PASSED with warnings:")
        for w in warnings:
            logger.warning(f"  WARNING: {w}")
    else:
        logger.info("Sanity check PASSED")

    return True


# ============== Main ==============
def main():
    setup_logging()
    logger = logging.getLogger('transform')

    engine = get_engine()

    # Шаг 2
    logger.info("=" * 60)
    logger.info("STEP 2: Transforming merged -> standart")
    logger.info("=" * 60)
    start = datetime.now()
    run_transform(engine)
    logger.info(f"Step 2 complete in {(datetime.now() - start).total_seconds():.1f}s")

    # Sanity check standart
    logger.info("-" * 60)
    sanity_check_standart(engine)

    # Шаг 3
    logger.info("=" * 60)
    logger.info("STEP 3: Tracking price history")
    logger.info("=" * 60)
    start = datetime.now()
    run_price_tracking(engine)
    logger.info(f"Step 3 complete in {(datetime.now() - start).total_seconds():.1f}s")

    # Sanity check price history
    logger.info("-" * 60)
    sanity_check_price_history(engine)


if __name__ == '__main__':
    main()