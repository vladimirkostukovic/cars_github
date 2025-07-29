import json
import re
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger("PriceTracker")

cfg = json.load(open(Path(__file__).parent / 'config.json'))
engine = create_engine(
    f"postgresql+psycopg2://{cfg['USER']}:{cfg['PWD']}@{cfg['HOST']}:{cfg['PORT']}/{cfg['DB']}"
)
SCHEMA = 'acars'
PATTERN = 'sauto_%'

def list_snapshots():
    sql = text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = :s AND table_name LIKE :p"
    )
    rows = engine.execute(sql, s=SCHEMA, p=PATTERN).fetchall()
    snapshots = []
    for r in rows:
        t = r[0]
        m = re.search(r'sauto_(\d{2})(\d{2})(\d{4})', t)
        if m:
            snapshots.append((t, datetime(int(m[3]), int(m[2]), int(m[1]))))
        else:
            log.warning(f"Пропускаю таблицу без даты: {t}")
    snapshots.sort(key=lambda x: x[1])
    return [s[0] for s in snapshots]
def ensure_price_history_table():
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

def track_price_changes_last_pair_nodup():
    snapshots = list_snapshots()
    if len(snapshots) < 2:
        log.warning("Недостаточно снапшотов для сравнения")
        return

    ensure_price_history_table()

    prev_snap, last_snap = snapshots[-2], snapshots[-1]
    log.info(f"Сравниваю {prev_snap} и {last_snap}")

    df_prev = pd.read_sql(text(f'SELECT id, price FROM {SCHEMA}."{prev_snap}"'), engine)
    df_last = pd.read_sql(text(f'SELECT id, price FROM {SCHEMA}."{last_snap}"'), engine)
    date_last = datetime.strptime(last_snap[-8:], '%d%m%Y').date()

    merged = df_last.merge(df_prev, on='id', how='inner', suffixes=('_new', '_old'))

    changes = merged[
        (merged['price_new'] != merged['price_old']) &
        (abs(merged['price_new'] - merged['price_old']) >= 1000)
    ].copy()

    if not changes.empty:
        changes['change_date'] = date_last
        changes = changes[['id', 'price_old', 'price_new', 'change_date']]
        changes = changes.rename(columns={'price_old': 'old_price', 'price_new': 'new_price'})
 # --- Антидубль: грузим price_history за сегодняшнюю дату, фильтруем то что уже есть ---
        try:
            today_hist = pd.read_sql(
                f"SELECT id, change_date FROM {SCHEMA}.price_history WHERE change_date = %(dt)s",
                engine, params={"dt": date_last}
            )
            mask = ~changes.set_index(['id', 'change_date']).index.isin(
                today_hist.set_index(['id', 'change_date']).index
            )
            changes = changes[mask]
        except Exception as e:
            log.warning(f"Не удалось загрузить price_history для дедупликации: {e}")

        if not changes.empty:
            changes.to_sql(
                'price_history',
                engine,
                schema=SCHEMA,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=5000
            )
            log.info(f"Внесено новых изменений цен: {len(changes)}")
        else:
            log.info("Все изменения уже записаны сегодня, ничего не добавлено")
    else:
        log.info("Изменений цен не найдено")

if __name__ == '__main__':
    track_price_changes_last_pair_nodup()