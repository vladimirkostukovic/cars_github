import pandas as pd
import numpy as np
import json
from sqlalchemy import create_engine
from pathlib import Path
from datetime import datetime

DEBUG = True

# === Конфиг и engine ===
cfg = json.load(open(Path(__file__).parent / 'config.json'))
engine = create_engine(
    f"postgresql+psycopg2://{cfg['USER']}:{cfg['PWD']}@{cfg['HOST']}:{cfg['PORT']}/{cfg['DB']}"
)
SCHEMA = 'acars'
TABLE = 'standart'
RECO_DISCOUNT = 0.9
RECO_FLAT = 13000
MIN_SALES_PER_GROUP = 5

# === 1. Грузим данные и метаданные ===
df = pd.read_sql(f'SELECT * FROM {SCHEMA}.{TABLE}', engine)

if DEBUG:
    print(f"[LOAD] Records: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"[LOAD] Missing 'id': {df['id'].isnull().sum()}, Duplicates: {df['id'].duplicated().sum()}")

# === Удаление дублей по id ===
dupes = df[df.duplicated('id', keep=False)].sort_values('id')
if not dupes.empty:
    print(f"[WARN] Удаляется {dupes.shape[0]} дублированных записей по 'id'")
    df = df.sort_values('dated_added').drop_duplicates('id', keep='last')

assert df['id'].is_unique, "❌ ID не уникальны"

if 'rok_vyroby' not in df.columns:
    df['rok_vyroby'] = pd.to_datetime(df['manuf_date'], errors='coerce').dt.year

# Фильтры по цене и году
df = df[(df['price'] >= 30000) & (df['price'] <= 5_000_000)]
df = df[(df['rok_vyroby'] >= 2015) & (df['rok_vyroby'] <= 2024)]
if DEBUG:
    print(f"[FILTER] Records after price/year filter: {df.shape[0]}")

# Дата актуальности
today = pd.Timestamp(datetime.now().date())
yesterday = today - pd.Timedelta(days=1)
SNAPSHOT_DATE = yesterday
print(f"[INFO] Построение аналитики по данным на: {SNAPSHOT_DATE.date()}")

def valid_date_sold(val):
    if pd.isnull(val):
        return None
    try:
        dt = pd.to_datetime(val)
        if isinstance(dt, pd.Timestamp):
            dt = dt.normalize()
        return dt if dt <= SNAPSHOT_DATE else None
    except Exception:
        return None

df['dated_added'] = pd.to_datetime(df['dated_added'], errors='coerce')
df = df[df['dated_added'] <= SNAPSHOT_DATE]  # Отсечка "будущих" лотов

df['date_sold_valid'] = df['date_sold'].apply(valid_date_sold)
df['days_on_market'] = (
    pd.to_datetime(df['date_sold_valid'], errors='coerce') - df['dated_added']
).dt.days
df['days_on_sale_now'] = (today - df['dated_added']).dt.days
df.loc[df['date_sold_valid'].notnull(), 'days_on_sale_now'] = None

invalid_sold_dates = df['date_sold'].notnull() & df['date_sold_valid'].isnull()
if DEBUG and invalid_sold_dates.sum() > 0:
    print(f"[WARN] Invalid sold dates skipped: {invalid_sold_dates.sum()}")
    print(df.loc[invalid_sold_dates, ['id', 'date_sold', 'dated_added']].sort_values('date_sold').head(10))
    df.loc[invalid_sold_dates, ['id', 'date_sold', 'dated_added']].to_csv('invalid_date_sold_debug.csv', index=False)

def skupina_tachometru(x):
    try: km = float(x)
    except Exception: return 'neznámý'
    if km < 10000: return 'do 10 000 km'
    if km < 50000: return '10 000–50 000 km'
    if km < 100000: return '50 000–100 000 km'
    if km < 150000: return '100 000–150 000 km'
    if km < 200000: return '150 000–200 000 km'
    if km < 300000: return '200 000–300 000 km'
    return 'nad 300 000 km'

df['skupina_tachometru'] = df['tachometr'].apply(skupina_tachometru)

group_cols = ['manufacture', 'model', 'rok_vyroby', 'fuel', 'gearbox', 'skupina_tachometru']

# === 2. Основная агрегация ===
agg = (
    df.groupby(group_cols)['price']
    .agg(['min', 'mean', 'max', 'count'])
    .reset_index()
    .rename(columns={'min': 'min_price', 'mean': 'avg_price', 'max': 'max_price', 'count': 'total_count'})
)
agg['cat1_limit'] = ((agg['min_price'] + (agg['max_price'] - agg['min_price']) / 3).round(0)).astype(int)
agg['cat2_limit'] = ((agg['min_price'] + 2 * (agg['max_price'] - agg['min_price']) / 3).round(0)).astype(int)

df = df.merge(agg[group_cols + ['cat1_limit', 'cat2_limit']], on=group_cols, how='left')

def price_cat(x, cat1, cat2):
    if x <= cat1: return 1
    elif x <= cat2: return 2
    else: return 3

df['price_cat'] = df.apply(lambda r: price_cat(r['price'], r['cat1_limit'], r['cat2_limit']), axis=1)
if DEBUG:
    print("[PRICE_CAT] Распределение:")
    print(df['price_cat'].value_counts(dropna=False).sort_index())

# === 3. Категориальные продажи и базовая статистика ===
sold = df[df['date_sold_valid'].notnull()]
sold_cat_counts = (
    sold.groupby(group_cols + ['price_cat'])['id']
    .count()
    .unstack(fill_value=0)
    .reset_index()
    .rename(columns={1: 'sold_in_cat1', 2: 'sold_in_cat2', 3: 'sold_in_cat3'})
)
for col in ['sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3']:
    sold_cat_counts[f'percent_{col}'] = (
        sold_cat_counts.get(col, 0) / (
            sold_cat_counts.get('sold_in_cat1', 0) +
            sold_cat_counts.get('sold_in_cat2', 0) +
            sold_cat_counts.get('sold_in_cat3', 0)
        ).replace(0, np.nan) * 100
    ).round(2).fillna(0)

sold_total = (
    sold.groupby(group_cols)['id']
    .count()
    .reset_index()
    .rename(columns={'id': 'total_sold'})
)

base_aggs = df.groupby(group_cols).agg(
    avg_mileage=('tachometr', 'mean'),
    avg_days_on_market=('days_on_market', 'mean'),
    avg_days_on_sale_now=('days_on_sale_now', 'mean')
).reset_index()

# === 4. Рекоммендованная цена ===
def reco_price(gr):
    low = gr[(gr['price_cat']==1) & (gr['date_sold_valid'].notnull())]['price']
    mid = gr[(gr['price_cat']==2) & (gr['date_sold_valid'].notnull())]['price']
    cnt_low, cnt_mid = len(low), len(mid)
    if cnt_low > 0 and cnt_low >= cnt_mid and cnt_low >= MIN_SALES_PER_GROUP:
        v = low.mean()
    elif cnt_mid > 0 and cnt_mid > cnt_low and cnt_mid >= MIN_SALES_PER_GROUP:
        v = np.mean([low.mean(), mid.mean()])
    else:
        return np.nan
    return max(v * RECO_DISCOUNT - RECO_FLAT, 0)

reco_df = df.groupby(group_cols).apply(reco_price).reset_index().rename(columns={0: 'recommended_price'})

# === 5. Ликвидность за 30 дней ===
cutoff = today - pd.Timedelta(days=30)
last_30 = df[df['dated_added'] >= cutoff]
sold_last_30 = last_30[last_30['date_sold_valid'].notnull()]
active_last_30 = last_30[last_30['date_sold_valid'].isnull()]
total_last_30 = last_30.groupby(group_cols)['id'].count().reset_index(name='total_count_30d')

liq_30 = (
    sold_last_30.groupby(group_cols)['id'].count().rename('sold_30d').to_frame()
    .join(active_last_30.groupby(group_cols)['id'].count().rename('active_30d'), how='outer')
    .fillna(0)
    .reset_index()
)
liq_30 = liq_30.merge(total_last_30, on=group_cols, how='left')
liq_30['sell_ratio_30d'] = liq_30['sold_30d'] / liq_30['total_count_30d'].replace(0, np.nan)

# === 6. История изменения цен ===
hist = pd.read_sql('SELECT id, old_price, new_price, change_date FROM acars.price_history', engine)
ph_sold = df[df['date_sold_valid'].notnull()][['id'] + group_cols]
hist = hist[hist['id'].isin(ph_sold['id'])]
def ph_metrics(sub):
    sub = sub.sort_values('change_date')
    start, end = sub.iloc[0]['old_price'], sub.iloc[-1]['new_price']
    drop_abs = start - end
    drop_pct = (drop_abs / start * 100) if start else np.nan
    return pd.Series({'price_drop_abs': drop_abs, 'price_drop_pct': drop_pct, 'n_price_changes': len(sub)})
ph_by_id = hist.groupby('id').apply(ph_metrics).reset_index()
ph_by_id = ph_by_id.merge(ph_sold, on='id', how='left')
ph_agg = ph_by_id.groupby(group_cols).agg(
    avg_price_drop_abs=('price_drop_abs', 'mean'),
    avg_price_drop_pct=('price_drop_pct', 'mean'),
    avg_price_changes=('n_price_changes', 'mean')
).reset_index()

# === 7. Финальный merge ===
agg_final = agg.merge(sold_cat_counts, on=group_cols, how='left') \
    .merge(sold_total, on=group_cols, how='left') \
    .merge(base_aggs, on=group_cols, how='left') \
    .merge(reco_df, on=group_cols, how='left') \
    .merge(liq_30, on=group_cols, how='left') \
    .merge(ph_agg, on=group_cols, how='left')

# === 8. Контроль и дополнительные фичи ===
agg_final.fillna({
    'sold_in_cat1': 0, 'sold_in_cat2': 0, 'sold_in_cat3': 0,
    'percent_sold_in_cat1': 0, 'percent_sold_in_cat2': 0, 'percent_sold_in_cat3': 0,
    'total_sold': 0, 'recommended_price': 0,
    'sold_30d': 0, 'active_30d': 0, 'total_count_30d': 0
}, inplace=True)

for col in ['sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3', 'total_sold', 'sold_30d', 'active_30d', 'total_count_30d']:
    agg_final[col] = agg_final[col].astype(int)

def calc_max_buy(row):
    min_price = row['min_price']
    max_price = row['max_price']
    cat1 = row['cat1_limit']
    cat2 = row['cat2_limit']
    s1 = row.get('sold_in_cat1', 0)
    s2 = row.get('sold_in_cat2', 0)
    s3 = row.get('sold_in_cat3', 0)
    if (s1 >= s2) and (s1 >= s3) and (s1 > 0):
        base = (min_price + cat1) / 2
    elif (s2 >= s1) and (s2 >= s3) and (s2 > 0):
        base = (cat1 + cat2) / 2
    elif s3 > 0:
        base = (cat2 + max_price) / 2
    else:
        base = min_price
    return round(base * 0.9, 0)

agg_final['max_buy_price'] = agg_final.apply(calc_max_buy, axis=1)

cat1_ads = df[df['price_cat'] == 1].copy()
cat1_ads['is_sold'] = cat1_ads['date_sold_valid'].notnull()
agg_cat1 = cat1_ads.groupby(group_cols).agg(
    total_in_cat1=('id', 'count'),
    sold_in_cat1=('is_sold', 'sum'),
    avg_days_in_cat1=('days_on_market', lambda x: x[cat1_ads.loc[x.index, 'is_sold']].mean() if len(x[cat1_ads.loc[x.index, 'is_sold']]) > 0 else 0)
).reset_index()

agg_final = agg_final.merge(agg_cat1, on=group_cols, how='left')
agg_final['total_in_cat1'] = agg_final['total_in_cat1'].fillna(0).astype(int)
agg_final['avg_days_in_cat1'] = agg_final['avg_days_in_cat1'].fillna(0)
agg_final['risk_percent'] = (100 - agg_final['percent_sold_in_cat1']).clip(0, 100)

def risk_with_penalty(row):
    penalty = 30 if row['avg_days_in_cat1'] > 60 else 10 if row['avg_days_in_cat1'] > 30 else 0
    return min(100, row['risk_percent'] + penalty)

agg_final['risk_score'] = agg_final.apply(risk_with_penalty, axis=1)
agg_final['low_risk_flag'] = (agg_final['percent_sold_in_cat1'] > 50) & (agg_final['avg_days_in_cat1'] < 30)
agg_final['risk_level'] = agg_final['low_risk_flag'].map({True: 'низкий', False: 'повышенный'})

# === 9. Финал ===
if DEBUG:
    print(f"[MERGE] agg_final rows: {agg_final.shape[0]}")
    print("[MERGE] NaN check:")
    print(agg_final.isnull().sum().sort_values(ascending=False).head(10))
    assert agg_final['recommended_price'].ge(0).all(), "❌ recommended_price < 0"
    assert agg_final['risk_score'].between(0, 100).all(), "❌ risk_score вне диапазона"

agg_final['snapshot_date'] = SNAPSHOT_DATE
# === DEBUG sanity checks по метрикам ===
if DEBUG:
    cols_to_check = [
        'recommended_price', 'risk_score', 'avg_price_drop_abs', 'avg_price_drop_pct',
        'avg_price_changes', 'sold_30d', 'active_30d', 'sell_ratio_30d',
        'total_sold', 'total_in_cat1', 'avg_days_in_cat1'
    ]

    print("\n[SANITY CHECK] Кол-во групп с NaN по ключевым метрикам:")
    for col in cols_to_check:
        n_nan = agg_final[col].isnull().sum()
        print(f"  - {col:22s}: {n_nan:6d} NaN")

    print("\n[SANITY CHECK] Распределение активности по группам:")
    print("  - Всего групп:", agg_final.shape[0])
    print("  - Групп с total_sold == 0:", (agg_final['total_sold'] == 0).sum())
    print("  - Групп с total_in_cat1 == 0:", (agg_final['total_in_cat1'] == 0).sum())
    print("  - Групп без продаж 30d:", (agg_final['sold_30d'] == 0).sum())
    print("  - Групп без активных лотов 30d:", (agg_final['active_30d'] == 0).sum())
    print("  - Групп без total_count_30d:", agg_final.get('total_count_30d', pd.Series(dtype=int)).isnull().sum())

    print("\n[SANITY CHECK] Диапазоны по числовым полям:")
    print(agg_final[[
        'avg_price_drop_pct', 'avg_price_drop_abs',
        'recommended_price', 'risk_score',
        'avg_days_on_market', 'avg_days_on_sale_now',
        'sold_30d', 'active_30d'
    ]].describe().T.round(1))
agg_final.to_sql('agg_price_cats', engine, schema=SCHEMA, if_exists='replace', index=False, method='multi')
print('[DONE] agg_price_cats обновлена.')