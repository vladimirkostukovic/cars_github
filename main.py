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
log = logging.getLogger(__name__)

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

def ensure_merged_table(df):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.merged (
        internal_id SERIAL PRIMARY KEY,
        {" ,".join([f'"{col}" TEXT' for col in df.columns if col not in ['date_added', 'date_sold']])},
        date_added DATE,
        date_sold DATE
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def load_merged():
    # Загружаем id и date_sold
    sql = text(f'SELECT id, date_sold FROM {SCHEMA}.merged')
    df = pd.read_sql(sql, engine)
    known_ids = set(df['id'])
    active_ids = set(df[df['date_sold'].isnull()]['id'])
    return known_ids, active_ids

def process_snapshot(snapshot, known_ids, active_ids):
    # Загружаем снапшот
    df = pd.read_sql(text(f'SELECT * FROM {SCHEMA}."{snapshot}"'), engine)
    snap_date = datetime.strptime(snapshot[-8:], '%d%m%Y')

    # сериализация json колонок
    json_cols = [
        'category_data', 'fuel_data', 'gearbox_data', 'locality_data',
        'manufacturer_data', 'model_data', 'premise_data', 'user_data'
    ]
    for col in json_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (dict, list)) else (None if pd.isnull(x) else x)
            )

    current_ids = set(df['id'])
    new_ids = current_ids - known_ids
    removed_ids = active_ids - current_ids

    # --- Добавляем новые ---
    new_count = 0
    if new_ids:
        new_df = df[df['id'].isin(new_ids)].copy()
        new_df['date_added'] = snap_date
        new_df['date_sold'] = None
        ensure_merged_table(new_df)
        new_df.to_sql('merged', engine, schema=SCHEMA, if_exists='append', index=False, method='multi', chunksize=20000)
        new_count = len(new_df)
        log.info(f"{snapshot}: добавлено новых объявлений: {new_count}")
    else:
        log.info(f"{snapshot}: новых объявлений нет")

    # --- Обновляем снятые ---
    sold_count = 0
    if removed_ids:
        upd = text(
            f"UPDATE {SCHEMA}.merged SET date_sold = :sold_date "
            "WHERE id = ANY(:ids) AND (date_sold IS NULL OR date_sold > :sold_date)"
        )
        with engine.begin() as conn:
            conn.execute(upd, {'sold_date': snap_date, 'ids': list(removed_ids)})
        sold_count = len(removed_ids)
        log.info(f"{snapshot}: снято с продажи (date_sold обновлён): {sold_count}")
    else:
        log.info(f"{snapshot}: снятых с продажи нет")

    # --- Обнуляем date_sold, если объявление вернулось ---
    reactivated_count = 0
    if current_ids:
        # Для числовых id (если текст — см. коммент ниже)
        ids_str = ','.join(map(str, current_ids))
        # ids_str = ','.join([f"'{x}'" for x in current_ids])  # если id строковые

        df_merged = pd.read_sql(
            f"SELECT id, date_sold FROM {SCHEMA}.merged WHERE id IN ({ids_str})",
            engine
        )
        ids_was_sold = set(df_merged[df_merged['date_sold'].notnull()]['id'])
        reactivated_ids = current_ids & ids_was_sold

        if reactivated_ids:
            sql = text(f"""
                UPDATE {SCHEMA}.merged
                SET date_sold = NULL
                WHERE id = ANY(:ids) AND date_sold IS NOT NULL
            """)
            with engine.begin() as conn:
                conn.execute(sql, {'ids': list(reactivated_ids)})
            reactivated_count = len(reactivated_ids)
            log.info(f"{snapshot}: объявлений вернулось в продажу, date_sold обнулён: {reactivated_count}")
        else:
            log.info(f"{snapshot}: не найдено объявлений, вернувшихся в продажу")
    else:
        log.info(f"{snapshot}: не найдено объявлений, вернувшихся в продажу")

    # Обновляем списки
    known_ids |= current_ids
    active_ids = (active_ids | new_ids | (reactivated_ids if current_ids else set())) - removed_ids

    return known_ids, active_ids, new_count, sold_count, reactivated_count
def main():
    snapshots = list_snapshots()
    if not snapshots:
        log.warning("Нет снапшотов")
        return

    log.info(f"Всего снапшотов: {len(snapshots)}. Запускаю обработку по порядку...")

    known_ids, active_ids = load_merged()
    new_total = 0
    sold_total = 0
    reactivated_total = 0

    for idx, snapshot in enumerate(snapshots, 1):
        log.info(f"[{idx}/{len(snapshots)}] --- {snapshot} ---")
        known_ids, active_ids, new_count, sold_count, reactivated_count = process_snapshot(snapshot, known_ids, active_ids)
        log.info(f"[{snapshot}] Итог: новых={new_count}, снятых={sold_count}, вернулось={reactivated_count}")
        new_total += new_count
        sold_total += sold_count
        reactivated_total += reactivated_count

    log.info("✔️ Обработка завершена.")
    log.info(f"Итого: новых добавлено {new_total}, снятых с продажи {sold_total}, вернулось {reactivated_total}")

if __name__ == '__main__':
    main()