import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
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

GROUP_COLS = ['manufacture', 'model', 'rok_vyroby', 'fuel', 'gearbox', 'skupina_tachometru']


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
def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce')


def safe_get_column(df: pd.DataFrame, col: str, default=0) -> pd.Series:
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def categorize_mileage(x) -> str:
    try:
        km = float(x)
    except (TypeError, ValueError):
        return 'unknown'

    if pd.isna(km):
        return 'unknown'
    if km < 10000:
        return 'to_10k'
    elif km < 50000:
        return '10k_50k'
    elif km < 100000:
        return '50k_100k'
    elif km < 150000:
        return '100k_150k'
    elif km < 200000:
        return '150k_200k'
    elif km < 300000:
        return '200k_300k'
    else:
        return 'over_300k'


def price_cat(price, cat1, cat2) -> int:
    try:
        price = float(price)
        cat1 = float(cat1)
        cat2 = float(cat2)
    except (TypeError, ValueError):
        return 2  # default middle category

    if pd.isna(price) or pd.isna(cat1) or pd.isna(cat2):
        return 2

    if price <= cat1:
        return 1
    elif price <= cat2:
        return 2
    else:
        return 3


# ============== Загрузка и очистка ==============
def load_standart(engine) -> pd.DataFrame:
    logger = logging.getLogger('analytics')
    df = pd.read_sql(f'SELECT * FROM {SCHEMA}.standart', engine)
    logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
    return df


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    # Числовые колонки
    numeric_cols = ['price', 'tachometr', 'rok_vyroby', 'power']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    # Даты
    date_cols = ['dated_added', 'date_sold', 'manuf_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # ID как int
    if 'id' in df.columns:
        df['id'] = safe_numeric(df['id']).astype('Int64')

    return df


def clean_data(df: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    # Конвертируем типы
    df = convert_types(df)

    # Дубликаты
    dupes = df[df.duplicated('id', keep=False)]
    if not dupes.empty:
        logger.warning(f"Removing {len(dupes)} duplicates")
        df = df.sort_values('dated_added').drop_duplicates('id', keep='last')

    # Год выпуска
    if 'rok_vyroby' not in df.columns or df['rok_vyroby'].isna().all():
        df['rok_vyroby'] = pd.to_datetime(df['manuf_date'], errors='coerce').dt.year

    # Убираем NaN перед фильтрацией
    df = df.dropna(subset=['price', 'rok_vyroby'])

    # Фильтры
    df = df[(df['price'] >= MIN_PRICE) & (df['price'] <= MAX_PRICE)]
    df = df[(df['rok_vyroby'] >= MIN_YEAR) & (df['rok_vyroby'] <= MAX_YEAR)]

    logger.info(f"After filters: {df.shape[0]} rows")
    return df.reset_index(drop=True)


def prepare_dates(df: pd.DataFrame, snapshot_date) -> pd.DataFrame:
    logger = logging.getLogger('analytics')
    today = pd.Timestamp(datetime.now().date())

    df['dated_added'] = pd.to_datetime(df['dated_added'], errors='coerce')
    df = df[df['dated_added'] <= snapshot_date]

    def valid_date_sold(val):
        if pd.isnull(val):
            return None
        try:
            dt = pd.to_datetime(val)
            if isinstance(dt, pd.Timestamp):
                dt = dt.normalize()
            return dt if dt <= snapshot_date else None
        except Exception:
            return None

    df['date_sold_valid'] = df['date_sold'].apply(valid_date_sold)

    # Расчёт дней
    sold_dates = pd.to_datetime(df['date_sold_valid'], errors='coerce')
    df['days_on_market'] = (sold_dates - df['dated_added']).dt.days
    df['days_on_sale_now'] = (today - df['dated_added']).dt.days
    df.loc[df['date_sold_valid'].notnull(), 'days_on_sale_now'] = np.nan

    return df.reset_index(drop=True)


def add_mileage_groups(df: pd.DataFrame) -> pd.DataFrame:
    df['skupina_tachometru'] = df['tachometr'].apply(categorize_mileage)
    return df


# ============== Агрегации ==============
def calc_basic_agg(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    agg = (df.groupby(GROUP_COLS, dropna=False)['price']
           .agg(['min', 'mean', 'max', 'count'])
           .reset_index()
           .rename(columns={'min': 'min_price', 'mean': 'avg_price',
                            'max': 'max_price', 'count': 'total_count'}))

    # Безопасный расчёт лимитов
    price_range = agg['max_price'] - agg['min_price']
    agg['cat1_limit'] = (agg['min_price'] + price_range / 3).round(0).fillna(0).astype(int)
    agg['cat2_limit'] = (agg['min_price'] + 2 * price_range / 3).round(0).fillna(0).astype(int)

    logger.info(f"Basic agg: {len(agg)} groups")
    return agg


def add_price_cats(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    df = df.merge(agg[GROUP_COLS + ['cat1_limit', 'cat2_limit']], on=GROUP_COLS, how='left')

    # Заполняем пропуски дефолтами
    df['cat1_limit'] = df['cat1_limit'].fillna(df['price'].quantile(0.33))
    df['cat2_limit'] = df['cat2_limit'].fillna(df['price'].quantile(0.66))

    df['price_cat'] = df.apply(
        lambda r: price_cat(r['price'], r['cat1_limit'], r['cat2_limit']),
        axis=1
    )

    logger.info(f"Price categories: {df['price_cat'].value_counts().to_dict()}")
    return df


def calc_cat_sales(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    sold = df[df['date_sold_valid'].notnull()].copy()

    if sold.empty:
        logger.warning("No sold records for cat_sales")
        result = df[GROUP_COLS].drop_duplicates().copy()
        for col in ['sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3',
                    'percent_sold_in_cat1', 'percent_sold_in_cat2', 'percent_sold_in_cat3']:
            result[col] = 0
        return result

    sold_cat = (sold.groupby(GROUP_COLS + ['price_cat'], dropna=False)['id']
                .count()
                .unstack(fill_value=0)
                .reset_index())

    # Переименовываем колонки безопасно
    rename_map = {1: 'sold_in_cat1', 2: 'sold_in_cat2', 3: 'sold_in_cat3'}
    for old, new in rename_map.items():
        if old in sold_cat.columns:
            sold_cat = sold_cat.rename(columns={old: new})
        else:
            sold_cat[new] = 0

    # Расчёт процентов
    for col in ['sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3']:
        sold_cat[col] = safe_get_column(sold_cat, col, 0)

    total = sold_cat['sold_in_cat1'] + sold_cat['sold_in_cat2'] + sold_cat['sold_in_cat3']
    total = total.replace(0, np.nan)

    sold_cat['percent_sold_in_cat1'] = (sold_cat['sold_in_cat1'] / total * 100).round(2).fillna(0)
    sold_cat['percent_sold_in_cat2'] = (sold_cat['sold_in_cat2'] / total * 100).round(2).fillna(0)
    sold_cat['percent_sold_in_cat3'] = (sold_cat['sold_in_cat3'] / total * 100).round(2).fillna(0)

    return sold_cat


def calc_total_sold(df: pd.DataFrame) -> pd.DataFrame:
    sold = df[df['date_sold_valid'].notnull()]

    if sold.empty:
        result = df[GROUP_COLS].drop_duplicates().copy()
        result['total_sold'] = 0
        return result

    return (sold.groupby(GROUP_COLS, dropna=False)['id']
            .count()
            .reset_index()
            .rename(columns={'id': 'total_sold'}))


def calc_base_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(GROUP_COLS, dropna=False)
            .agg(
                avg_mileage=('tachometr', 'mean'),
                avg_days_on_market=('days_on_market', 'mean'),
                avg_days_on_sale_now=('days_on_sale_now', 'mean')
            )
            .reset_index())


def calc_reco_price(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    def reco(gr):
        sold = gr[gr['date_sold_valid'].notnull()]
        low = sold[sold['price_cat'] == 1]['price']
        mid = sold[sold['price_cat'] == 2]['price']

        cnt_low = len(low)
        cnt_mid = len(mid)

        if cnt_low > 0 and cnt_low >= cnt_mid and cnt_low >= MIN_SALES:
            v = low.mean()
        elif cnt_mid > 0 and cnt_mid > cnt_low and cnt_mid >= MIN_SALES:
            low_mean = low.mean() if cnt_low > 0 else mid.mean()
            v = (low_mean + mid.mean()) / 2
        else:
            return np.nan

        result = v * RECO_DISCOUNT - RECO_FLAT
        return max(result, 0)

    result = (df.groupby(GROUP_COLS, dropna=False)
              .apply(reco, include_groups=False)
              .reset_index()
              .rename(columns={0: 'recommended_price'}))

    return result


def calc_liq_30d(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    today = pd.Timestamp(datetime.now().date())
    cutoff = today - pd.Timedelta(days=LIQ_DAYS)

    last_30 = df[df['dated_added'] >= cutoff].copy()

    if last_30.empty:
        logger.warning("No data in last 30 days")
        result = df[GROUP_COLS].drop_duplicates().copy()
        result['sold_30d'] = 0
        result['active_30d'] = 0
        result['total_count_30d'] = 0
        result['sell_ratio_30d'] = np.nan
        return result

    sold_30 = last_30[last_30['date_sold_valid'].notnull()]
    active_30 = last_30[last_30['date_sold_valid'].isnull()]

    # Total
    total_30 = (last_30.groupby(GROUP_COLS, dropna=False)['id']
                .count()
                .reset_index(name='total_count_30d'))

    # Sold
    if not sold_30.empty:
        sold_agg = (sold_30.groupby(GROUP_COLS, dropna=False)['id']
                    .count()
                    .reset_index(name='sold_30d'))
    else:
        sold_agg = df[GROUP_COLS].drop_duplicates().copy()
        sold_agg['sold_30d'] = 0

    # Active
    if not active_30.empty:
        active_agg = (active_30.groupby(GROUP_COLS, dropna=False)['id']
                      .count()
                      .reset_index(name='active_30d'))
    else:
        active_agg = df[GROUP_COLS].drop_duplicates().copy()
        active_agg['active_30d'] = 0

    # Merge
    liq = total_30.merge(sold_agg, on=GROUP_COLS, how='left')
    liq = liq.merge(active_agg, on=GROUP_COLS, how='left')

    liq['sold_30d'] = liq['sold_30d'].fillna(0)
    liq['active_30d'] = liq['active_30d'].fillna(0)

    # Ratio
    liq['sell_ratio_30d'] = liq['sold_30d'] / liq['total_count_30d'].replace(0, np.nan)

    return liq


def calc_price_hist_metrics(engine, df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    try:
        hist = pd.read_sql(f'SELECT id, old_price, new_price, change_date FROM {SCHEMA}.price_history', engine)
    except Exception as e:
        logger.warning(f"Could not load price_history: {e}")
        result = df[GROUP_COLS].drop_duplicates().copy()
        result['avg_price_drop_abs'] = np.nan
        result['avg_price_drop_pct'] = np.nan
        result['avg_price_changes'] = np.nan
        return result

    if hist.empty:
        logger.info("price_history is empty")
        result = df[GROUP_COLS].drop_duplicates().copy()
        result['avg_price_drop_abs'] = np.nan
        result['avg_price_drop_pct'] = np.nan
        result['avg_price_changes'] = np.nan
        return result

    # Конвертируем типы
    hist['id'] = safe_numeric(hist['id']).astype('Int64')
    hist['old_price'] = safe_numeric(hist['old_price'])
    hist['new_price'] = safe_numeric(hist['new_price'])
    hist['change_date'] = pd.to_datetime(hist['change_date'], errors='coerce')

    ph_sold = df[df['date_sold_valid'].notnull()][['id'] + GROUP_COLS].copy()
    hist = hist[hist['id'].isin(ph_sold['id'])]

    if hist.empty:
        logger.info("No price history for sold items")
        result = df[GROUP_COLS].drop_duplicates().copy()
        result['avg_price_drop_abs'] = np.nan
        result['avg_price_drop_pct'] = np.nan
        result['avg_price_changes'] = np.nan
        return result

    def metrics(sub):
        sub = sub.sort_values('change_date')
        start = sub.iloc[0]['old_price']
        end = sub.iloc[-1]['new_price']

        if pd.isna(start) or pd.isna(end) or start == 0:
            return pd.Series({
                'price_drop_abs': np.nan,
                'price_drop_pct': np.nan,
                'n_changes': len(sub)
            })

        drop_abs = start - end
        drop_pct = (drop_abs / start * 100)
        return pd.Series({
            'price_drop_abs': drop_abs,
            'price_drop_pct': drop_pct,
            'n_changes': len(sub)
        })

    ph_by_id = hist.groupby('id', dropna=False).apply(metrics, include_groups=False).reset_index()
    ph_by_id = ph_by_id.merge(ph_sold, on='id', how='left')

    result = (ph_by_id.groupby(GROUP_COLS, dropna=False)
              .agg(
                  avg_price_drop_abs=('price_drop_abs', 'mean'),
                  avg_price_drop_pct=('price_drop_pct', 'mean'),
                  avg_price_changes=('n_changes', 'mean')
              )
              .reset_index())

    return result


def calc_cat1_metrics(df: pd.DataFrame) -> pd.DataFrame:
    logger = logging.getLogger('analytics')

    cat1 = df[df['price_cat'] == 1].copy()

    if cat1.empty:
        logger.warning("No cat1 records")
        result = df[GROUP_COLS].drop_duplicates().copy()
        result['total_in_cat1'] = 0
        result['sold_in_cat1_detail'] = 0
        result['avg_days_in_cat1'] = 0
        return result

    cat1['is_sold'] = cat1['date_sold_valid'].notnull()

    def agg_cat1(gr):
        total = len(gr)
        sold_count = gr['is_sold'].sum()

        sold_days = gr.loc[gr['is_sold'], 'days_on_market']
        avg_days = sold_days.mean() if len(sold_days) > 0 else 0

        return pd.Series({
            'total_in_cat1': total,
            'sold_in_cat1_detail': sold_count,
            'avg_days_in_cat1': avg_days
        })

    result = (cat1.groupby(GROUP_COLS, dropna=False)
              .apply(agg_cat1, include_groups=False)
              .reset_index())

    return result


def calc_max_buy(row) -> float:
    try:
        min_p = float(row.get('min_price', 0) or 0)
        max_p = float(row.get('max_price', 0) or 0)
        c1 = float(row.get('cat1_limit', 0) or 0)
        c2 = float(row.get('cat2_limit', 0) or 0)
        s1 = int(row.get('sold_in_cat1', 0) or 0)
        s2 = int(row.get('sold_in_cat2', 0) or 0)
        s3 = int(row.get('sold_in_cat3', 0) or 0)
    except (TypeError, ValueError):
        return 0

    if (s1 >= s2) and (s1 >= s3) and (s1 > 0):
        base = (min_p + c1) / 2
    elif (s2 >= s1) and (s2 >= s3) and (s2 > 0):
        base = (c1 + c2) / 2
    elif s3 > 0:
        base = (c2 + max_p) / 2
    else:
        base = min_p

    return round(base * 0.9, 0)


def calc_risk(df: pd.DataFrame) -> pd.DataFrame:
    pct = safe_get_column(df, 'percent_sold_in_cat1', 0)
    avg_days = safe_get_column(df, 'avg_days_in_cat1', 0)

    df['risk_percent'] = (100 - pct).clip(0, 100)

    # Расчёт штрафа
    def risk_penalty(days, risk_pct):
        try:
            days = float(days) if not pd.isna(days) else 0
            risk_pct = float(risk_pct) if not pd.isna(risk_pct) else 100
        except (TypeError, ValueError):
            return 100

        if days > 60:
            penalty = 30
        elif days > 30:
            penalty = 10
        else:
            penalty = 0

        return min(100, risk_pct + penalty)

    df['risk_score'] = [risk_penalty(d, r) for d, r in zip(avg_days, df['risk_percent'])]

    # Low risk flag
    df['low_risk_flag'] = (pct > 50) & (avg_days < 30)
    df['risk_level'] = df['low_risk_flag'].map({True: 'low', False: 'high'})

    return df


def merge_metrics(agg: pd.DataFrame, *dfs) -> pd.DataFrame:
    result = agg.copy()
    for d in dfs:
        if d is not None and not d.empty:
            result = result.merge(d, on=GROUP_COLS, how='left')
    return result


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Числовые колонки с 0
    zero_cols = [
        'sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3',
        'percent_sold_in_cat1', 'percent_sold_in_cat2', 'percent_sold_in_cat3',
        'total_sold', 'recommended_price',
        'sold_30d', 'active_30d', 'total_count_30d',
        'total_in_cat1', 'sold_in_cat1_detail', 'avg_days_in_cat1'
    ]

    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Integer колонки
    int_cols = [
        'sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3', 'total_sold',
        'sold_30d', 'active_30d', 'total_count_30d', 'total_in_cat1',
        'sold_in_cat1_detail'
    ]

    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def validate_results(df: pd.DataFrame) -> bool:
    logger = logging.getLogger('analytics')

    errors = []

    if 'recommended_price' in df.columns:
        neg_prices = (df['recommended_price'] < 0).sum()
        if neg_prices > 0:
            errors.append(f"Negative recommended_price: {neg_prices}")

    if 'risk_score' in df.columns:
        bad_risk = ((df['risk_score'] < 0) | (df['risk_score'] > 100)).sum()
        if bad_risk > 0:
            errors.append(f"Risk score out of range: {bad_risk}")

    if errors:
        for e in errors:
            logger.error(e)
        return False

    logger.info(f"Validation passed. Final rows: {len(df)}")
    return True


def save_results(engine, df: pd.DataFrame, snapshot_date, table: str = 'agg_price_cats'):
    logger = logging.getLogger('analytics')
    df['snapshot_date'] = snapshot_date
    df.to_sql(table, engine, schema=SCHEMA,
              if_exists='replace', index=False, method='multi')
    logger.info(f'{table} updated with {len(df)} rows')


# ============== Main ==============
def run_analytics(engine):
    logger = logging.getLogger('analytics')
    logger.info("Starting analytics...")

    today = pd.Timestamp(datetime.now().date())
    snapshot_date = today - pd.Timedelta(days=1)
    logger.info(f"Analytics date: {snapshot_date.date()}")

    # Загрузка и подготовка
    df = load_standart(engine)
    df = clean_data(df, snapshot_date)
    df = prepare_dates(df, snapshot_date)
    df = add_mileage_groups(df)

    # Базовая агрегация
    agg = calc_basic_agg(df)
    df = add_price_cats(df, agg)

    # Расчёт метрик
    logger.info("Calculating metrics...")
    sold_cat = calc_cat_sales(df)
    sold_tot = calc_total_sold(df)
    base = calc_base_metrics(df)
    reco = calc_reco_price(df)
    liq = calc_liq_30d(df)
    ph = calc_price_hist_metrics(engine, df)

    # Объединение
    logger.info("Merging metrics...")
    final = merge_metrics(agg, sold_cat, sold_tot, base, reco, liq, ph)
    final = fill_missing(final)
    final['max_buy_price'] = final.apply(calc_max_buy, axis=1)

    # Cat1 метрики и риск
    cat1 = calc_cat1_metrics(df)
    final = final.merge(cat1, on=GROUP_COLS, how='left')
    final = fill_missing(final)
    final = calc_risk(final)

    # Валидация и сохранение
    if validate_results(final):
        save_results(engine, final, snapshot_date)
    else:
        logger.error("Validation failed, results not saved")

    logger.info("✓ Analytics complete")
    return final


def main():
    setup_logging()
    logger = logging.getLogger('analytics')
    logger.info("=" * 60)
    logger.info("STEP 4: Running analytics")
    logger.info("=" * 60)

    start = datetime.now()
    engine = get_engine()
    run_analytics(engine)
    logger.info(f"Step 4 complete in {(datetime.now() - start).total_seconds():.1f}s")


if __name__ == '__main__':
    main()