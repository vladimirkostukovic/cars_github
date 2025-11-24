import json
import re
import logging
from datetime import datetime
from typing import Set, Tuple, List

import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path


# ============== Константы ==============
SCHEMA = 'acars'
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


# ============== Сериализация ==============
def serialize_value(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False, default=str)
    return x


def serialize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('merge')

    for col in df.columns:
        # Пропускаем даты — их не трогаем
        if col in ['date_added', 'date_sold']:
            continue

        # Проверяем есть ли сложные типы
        sample = df[col].dropna().head(50)
        if len(sample) == 0:
            continue

        has_complex = any(isinstance(x, (dict, list)) for x in sample)

        if has_complex:
            logger.debug(f"Serializing column: {col}")
            df[col] = df[col].apply(serialize_value)

    return df


# ============== Отслеживание обработанных снапшотов ==============
def ensure_tracking_table(engine):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.processed_snapshots (
        snapshot_name TEXT PRIMARY KEY,
        processed_at TIMESTAMP DEFAULT NOW(),
        records_added INT,
        records_sold INT,
        records_reactivated INT
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def get_processed_snapshots(engine) -> Set[str]:
    try:
        df = pd.read_sql(f'SELECT snapshot_name FROM {SCHEMA}.processed_snapshots', engine)
        return set(df['snapshot_name'])
    except Exception:
        return set()


def mark_snapshot_processed(engine, name: str, added: int, sold: int, reactivated: int):
    sql = text(f"""
        INSERT INTO {SCHEMA}.processed_snapshots 
        (snapshot_name, records_added, records_sold, records_reactivated)
        VALUES (:name, :added, :sold, :reactivated)
        ON CONFLICT (snapshot_name) DO NOTHING
    """)
    with engine.begin() as conn:
        conn.execute(sql, {'name': name, 'added': added, 'sold': sold, 'reactivated': reactivated})


# ============== Работа со снапшотами ==============
def list_snapshots(engine) -> List[str]:
    logger = logging.getLogger('merge')
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
        else:
            logger.warning(f"Skipping table without date: {name}")

    snapshots.sort(key=lambda x: x[1])
    return [s[0] for s in snapshots]


def get_new_snapshots(engine) -> List[str]:
    all_snapshots = list_snapshots(engine)
    processed = get_processed_snapshots(engine)
    new = [s for s in all_snapshots if s not in processed]
    return new


def load_snapshot(engine, name: str) -> pd.DataFrame:
    logger = logging.getLogger('merge')
    df = pd.read_sql(text(f'SELECT * FROM {SCHEMA}."{name}"'), engine)
    logger.info(f"Loaded {name}: {len(df)} records")
    return df


def get_snapshot_date(name: str) -> datetime:
    return datetime.strptime(name[-8:], '%d%m%Y')


# ============== Операции с merged ==============
def ensure_merged_table(engine, df: pd.DataFrame):
    logger = logging.getLogger('merge')
    cols = [f'"{c}" TEXT' for c in df.columns if c not in ['date_added', 'date_sold']]
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.merged (
        internal_id SERIAL PRIMARY KEY,
        {", ".join(cols)},
        date_added DATE,
        date_sold DATE
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Merged table checked/created")


def load_merged_state(engine) -> Tuple[Set[int], Set[int]]:
    logger = logging.getLogger('merge')
    try:
        df = pd.read_sql(text(f'SELECT id, date_sold FROM {SCHEMA}.merged'), engine)
        df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
        known = set(df['id'].dropna().astype(int))
        active = set(df[df['date_sold'].isnull()]['id'].dropna().astype(int))
        logger.info(f"State loaded: known={len(known)}, active={len(active)}")
        return known, active
    except Exception as e:
        logger.warning(f"Could not load merged (may not exist): {e}")
        return set(), set()


def add_new_listings(engine, df: pd.DataFrame, ids: Set[int], date: datetime) -> int:
    logger = logging.getLogger('merge')
    if not ids:
        logger.info("No new listings")
        return 0

    new_df = df[df['id'].isin(ids)].copy()
    new_df['date_added'] = date
    new_df['date_sold'] = None

    ensure_merged_table(engine, new_df)
    new_df.to_sql('merged', engine, schema=SCHEMA,
                  if_exists='append', index=False, method='multi', chunksize=5000)

    logger.info(f"Added {len(new_df)} new listings")
    return len(new_df)


def mark_sold(engine, ids: Set[int], date: datetime) -> int:
    logger = logging.getLogger('merge')
    if not ids:
        logger.info("No sold listings")
        return 0

    sql = text(
        f"UPDATE {SCHEMA}.merged SET date_sold = :date "
        "WHERE id = ANY(:ids) AND (date_sold IS NULL OR date_sold > :date)"
    )
    with engine.begin() as conn:
        conn.execute(sql, {'date': date, 'ids': [str(i) for i in ids]})

    logger.info(f"Marked {len(ids)} as sold")
    return len(ids)


def reactivate_listings(engine, ids: Set[int]) -> int:
    logger = logging.getLogger('merge')
    if not ids:
        return 0

    ids_str = ','.join(f"'{i}'" for i in ids)
    df = pd.read_sql(
        f"SELECT id, date_sold FROM {SCHEMA}.merged WHERE id IN ({ids_str})",
        engine
    )

    was_sold = set(df[df['date_sold'].notnull()]['id'].astype(str))
    reactivated = {str(i) for i in ids} & was_sold

    if not reactivated:
        return 0

    sql = text(f"UPDATE {SCHEMA}.merged SET date_sold = NULL "
               f"WHERE id = ANY(:ids) AND date_sold IS NOT NULL")
    with engine.begin() as conn:
        conn.execute(sql, {'ids': list(reactivated)})

    logger.info(f"Reactivated {len(reactivated)} listings")
    return len(reactivated)


def get_reactivated_ids(engine, ids: Set[int]) -> Set[int]:
    if not ids:
        return set()
    ids_str = ','.join(f"'{i}'" for i in ids)
    df = pd.read_sql(
        f"SELECT id, date_sold FROM {SCHEMA}.merged WHERE id IN ({ids_str})",
        engine
    )
    was_sold = set(int(x) for x in df[df['date_sold'].notnull()]['id'])
    return ids & was_sold


def process_snapshot(engine, name: str, known: Set[int], active: Set[int]):
    logger = logging.getLogger('merge')

    df = load_snapshot(engine, name)
    date = get_snapshot_date(name)

    # Сериализовать все сложные типы
    df = serialize_dataframe(df)

    # id теперь может быть строкой — работаем с int
    df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')
    current = set(df['id'].dropna().astype(int))

    new_ids = current - known
    removed = active - current

    new_count = add_new_listings(engine, df, new_ids, date)
    sold_count = mark_sold(engine, removed, date)
    reactivated_count = reactivate_listings(engine, current)

    known |= current
    reactivated = get_reactivated_ids(engine, current) if current else set()
    active = (active | new_ids | reactivated) - removed

    return known, active, new_count, sold_count, reactivated_count


# ============== Main ==============
def run_merge(engine):
    logger = logging.getLogger('merge')

    ensure_tracking_table(engine)

    new_snapshots = get_new_snapshots(engine)

    if not new_snapshots:
        logger.info("No new snapshots to process")
        return

    logger.info(f"New snapshots to process: {len(new_snapshots)}")
    for s in new_snapshots:
        logger.info(f"  - {s}")

    known, active = load_merged_state(engine)
    total_new = total_sold = total_reactivated = 0

    for idx, snap in enumerate(new_snapshots, 1):
        logger.info(f"[{idx}/{len(new_snapshots)}] Processing {snap}")

        known, active, new, sold, reactivated = process_snapshot(engine, snap, known, active)

        mark_snapshot_processed(engine, snap, new, sold, reactivated)

        logger.info(f"[{snap}] Summary: new={new}, sold={sold}, reactivated={reactivated}")

        total_new += new
        total_sold += sold
        total_reactivated += reactivated

    logger.info("✓ Merge complete")
    logger.info(f"Total: new={total_new}, sold={total_sold}, reactivated={total_reactivated}")


def main():
    setup_logging()
    logger = logging.getLogger('merge')
    logger.info("=" * 60)
    logger.info("STEP 1: Processing snapshots -> merged")
    logger.info("=" * 60)

    start = datetime.now()
    engine = get_engine()
    run_merge(engine)
    elapsed = (datetime.now() - start).total_seconds()
    logger.info(f"Step 1 complete in {elapsed:.1f}s")


if __name__ == '__main__':
    main()