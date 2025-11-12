
import pandas as pd
import numpy as np
from datetime import datetime
from analytics_engine import AnalyticsEngine


class AnalyticsEngineExtended(AnalyticsEngine):
    
    def calc_cat_sales(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        sold = df[df['date_sold_valid'].notnull()]
        
        sold_cat = (sold.groupby(gc + ['price_cat'])['id'].count()
                   .unstack(fill_value=0).reset_index()
                   .rename(columns={1: 'sold_in_cat1', 2: 'sold_in_cat2', 3: 'sold_in_cat3'}))
        
        for col in ['sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3']:
            total = (sold_cat.get('sold_in_cat1', 0) + sold_cat.get('sold_in_cat2', 0) + 
                    sold_cat.get('sold_in_cat3', 0)).replace(0, np.nan)
            sold_cat[f'percent_{col}'] = (sold_cat.get(col, 0) / total * 100).round(2).fillna(0)
        
        return sold_cat
    
    def calc_total_sold(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        sold = df[df['date_sold_valid'].notnull()]
        return sold.groupby(gc)['id'].count().reset_index().rename(columns={'id': 'total_sold'})
    
    def calc_base_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        return df.groupby(gc).agg(
            avg_mileage=('tachometr', 'mean'),
            avg_days_on_market=('days_on_market', 'mean'),
            avg_days_on_sale_now=('days_on_sale_now', 'mean')
        ).reset_index()
    
    def calc_reco_price(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        
        def reco(gr):
            low = gr[(gr['price_cat']==1) & (gr['date_sold_valid'].notnull())]['price']
            mid = gr[(gr['price_cat']==2) & (gr['date_sold_valid'].notnull())]['price']
            cnt_low, cnt_mid = len(low), len(mid)
            
            if cnt_low > 0 and cnt_low >= cnt_mid and cnt_low >= self.MIN_SALES:
                v = low.mean()
            elif cnt_mid > 0 and cnt_mid > cnt_low and cnt_mid >= self.MIN_SALES:
                v = np.mean([low.mean(), mid.mean()])
            else:
                return np.nan
            return max(v * self.RECO_DISCOUNT - self.RECO_FLAT, 0)
        
        return df.groupby(gc).apply(reco).reset_index().rename(columns={0: 'recommended_price'})
    
    def calc_liq_30d(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        today = pd.Timestamp(datetime.now().date())
        cutoff = today - pd.Timedelta(days=self.LIQ_DAYS)
        
        last_30 = df[df['dated_added'] >= cutoff]
        sold_30 = last_30[last_30['date_sold_valid'].notnull()]
        active_30 = last_30[last_30['date_sold_valid'].isnull()]
        
        total_30 = last_30.groupby(gc)['id'].count().reset_index(name='total_count_30d')
        
        liq = (sold_30.groupby(gc)['id'].count().rename('sold_30d').to_frame()
               .join(active_30.groupby(gc)['id'].count().rename('active_30d'), how='outer')
               .fillna(0).reset_index())
        
        liq = liq.merge(total_30, on=gc, how='left')
        liq['sell_ratio_30d'] = liq['sold_30d'] / liq['total_count_30d'].replace(0, np.nan)
        return liq
    
    def calc_price_hist_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        hist = pd.read_sql('SELECT id, old_price, new_price, change_date FROM acars.price_history', self.engine)
        
        ph_sold = df[df['date_sold_valid'].notnull()][['id'] + gc]
        hist = hist[hist['id'].isin(ph_sold['id'])]
        
        def metrics(sub):
            sub = sub.sort_values('change_date')
            start, end = sub.iloc[0]['old_price'], sub.iloc[-1]['new_price']
            drop_abs = start - end
            drop_pct = (drop_abs / start * 100) if start else np.nan
            return pd.Series({'price_drop_abs': drop_abs, 'price_drop_pct': drop_pct, 'n_changes': len(sub)})
        
        ph_by_id = hist.groupby('id').apply(metrics).reset_index()
        ph_by_id = ph_by_id.merge(ph_sold, on='id', how='left')
        
        return ph_by_id.groupby(gc).agg(
            avg_price_drop_abs=('price_drop_abs', 'mean'),
            avg_price_drop_pct=('price_drop_pct', 'mean'),
            avg_price_changes=('n_changes', 'mean')
        ).reset_index()
    
    def calc_max_buy(self, row: pd.Series) -> float:
        min_p, max_p = row['min_price'], row['max_price']
        c1, c2 = row['cat1_limit'], row['cat2_limit']
        s1, s2, s3 = row.get('sold_in_cat1', 0), row.get('sold_in_cat2', 0), row.get('sold_in_cat3', 0)
        
        if (s1 >= s2) and (s1 >= s3) and (s1 > 0):
            base = (min_p + c1) / 2
        elif (s2 >= s1) and (s2 >= s3) and (s2 > 0):
            base = (c1 + c2) / 2
        elif s3 > 0:
            base = (c2 + max_p) / 2
        else:
            base = min_p
        return round(base * 0.9, 0)
    
    def calc_cat1_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        cat1 = df[df['price_cat'] == 1].copy()
        cat1['is_sold'] = cat1['date_sold_valid'].notnull()
        
        return cat1.groupby(gc).agg(
            total_in_cat1=('id', 'count'),
            sold_in_cat1=('is_sold', 'sum'),
            avg_days_in_cat1=('days_on_market', lambda x: x[cat1.loc[x.index, 'is_sold']].mean() 
                             if len(x[cat1.loc[x.index, 'is_sold']]) > 0 else 0)
        ).reset_index()
    
    def calc_risk(self, df: pd.DataFrame) -> pd.DataFrame:
        df['risk_percent'] = (100 - df['percent_sold_in_cat1']).clip(0, 100)
        
        def risk_penalty(row):
            penalty = 30 if row['avg_days_in_cat1'] > 60 else 10 if row['avg_days_in_cat1'] > 30 else 0
            return min(100, row['risk_percent'] + penalty)
        
        df['risk_score'] = df.apply(risk_penalty, axis=1)
        df['low_risk_flag'] = (df['percent_sold_in_cat1'] > 50) & (df['avg_days_in_cat1'] < 30)
        df['risk_level'] = df['low_risk_flag'].map({True: 'low', False: 'high'})
        return df
    
    def merge_metrics(self, agg: pd.DataFrame, *dfs: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        result = agg
        for d in dfs:
            result = result.merge(d, on=gc, how='left')
        return result
    
    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df.fillna({
            'sold_in_cat1': 0, 'sold_in_cat2': 0, 'sold_in_cat3': 0,
            'percent_sold_in_cat1': 0, 'percent_sold_in_cat2': 0, 'percent_sold_in_cat3': 0,
            'total_sold': 0, 'recommended_price': 0,
            'sold_30d': 0, 'active_30d': 0, 'total_count_30d': 0
        }, inplace=True)
        
        for col in ['sold_in_cat1', 'sold_in_cat2', 'sold_in_cat3', 'total_sold', 
                   'sold_30d', 'active_30d', 'total_count_30d']:
            df[col] = df[col].astype(int)
        
        df['total_in_cat1'] = df.get('total_in_cat1', 0).fillna(0).astype(int)
        df['avg_days_in_cat1'] = df.get('avg_days_in_cat1', 0).fillna(0)
        return df
    
    def validate(self, df: pd.DataFrame):
        assert df['recommended_price'].ge(0).all(), "Negative recommended price"
        assert df['risk_score'].between(0, 100).all(), "Risk score out of range"
        if self.debug:
            self.logger.info(f"Final rows: {df.shape[0]}")
    
    def save_results(self, df: pd.DataFrame, table: str = 'agg_price_cats'):
        df['snapshot_date'] = self.snapshot_date
        df.to_sql(table, self.engine, schema=self.schema, 
                 if_exists='replace', index=False, method='multi')
        self.logger.info(f'{table} updated')
    
    def run_full(self):
        self.logger.info("Starting analytics...")
        
        df = self.load_data()
        df = self.clean_data(df)
        df = self.prepare_dates(df)
        df = self.add_mileage_groups(df)
        
        agg = self.calc_basic_agg(df)
        df = self.add_price_cats(df, agg)
        
        sold_cat = self.calc_cat_sales(df)
        sold_tot = self.calc_total_sold(df)
        base = self.calc_base_metrics(df)
        reco = self.calc_reco_price(df)
        liq = self.calc_liq_30d(df)
        ph = self.calc_price_hist_metrics(df)
        
        final = self.merge_metrics(agg, sold_cat, sold_tot, base, reco, liq, ph)
        final = self.fill_missing(final)
        final['max_buy_price'] = final.apply(self.calc_max_buy, axis=1)
        
        cat1 = self.calc_cat1_metrics(df)
        final = final.merge(cat1, on=self.get_group_cols(), how='left')
        final = self.fill_missing(final)
        final = self.calc_risk(final)
        
        self.validate(final)
        self.save_results(final)
        
        self.logger.info("✓ Analytics complete")
        return final
