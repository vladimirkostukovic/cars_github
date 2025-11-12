
import logging
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
from sqlalchemy.engine import Engine


class AnalyticsEngine:
    # Constants
    RECO_DISCOUNT = 0.9
    RECO_FLAT = 13000
    MIN_SALES = 5
    MIN_PRICE = 30000
    MAX_PRICE = 5_000_000
    MIN_YEAR = 2015
    MAX_YEAR = 2024
    LIQ_DAYS = 30
    
    def __init__(self, engine: Engine, schema: str = 'acars', table: str = 'standart', debug: bool = False):
        self.engine = engine
        self.schema = schema
        self.table = table
        self.debug = debug
        self.logger = logging.getLogger(self.__class__.__name__)
        
        today = pd.Timestamp(datetime.now().date())
        self.snapshot_date = today - pd.Timedelta(days=1)
        self.logger.info(f"Analytics date: {self.snapshot_date.date()}")
    
    def load_data(self) -> pd.DataFrame:
        df = pd.read_sql(f'SELECT * FROM {self.schema}.{self.table}', self.engine)
        if self.debug:
            self.logger.info(f"Loaded: {df.shape[0]} rows, {df.shape[1]} cols")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove duplicates
        dupes = df[df.duplicated('id', keep=False)]
        if not dupes.empty:
            self.logger.warning(f"Removing {len(dupes)} duplicates")
            df = df.sort_values('dated_added').drop_duplicates('id', keep='last')
        
        assert df['id'].is_unique, "IDs not unique"
        
        # Add year
        if 'rok_vyroby' not in df.columns:
            df['rok_vyroby'] = pd.to_datetime(df['manuf_date'], errors='coerce').dt.year
        
        # Filters
        df = df[(df['price'] >= self.MIN_PRICE) & (df['price'] <= self.MAX_PRICE)]
        df = df[(df['rok_vyroby'] >= self.MIN_YEAR) & (df['rok_vyroby'] <= self.MAX_YEAR)]
        
        if self.debug:
            self.logger.info(f"After filters: {df.shape[0]} rows")
        return df
    
    def prepare_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        today = pd.Timestamp(datetime.now().date())
        
        df['dated_added'] = pd.to_datetime(df['dated_added'], errors='coerce')
        df = df[df['dated_added'] <= self.snapshot_date]
        
        df['date_sold_valid'] = df['date_sold'].apply(self._valid_date_sold)
        df['days_on_market'] = (pd.to_datetime(df['date_sold_valid'], errors='coerce') - df['dated_added']).dt.days
        df['days_on_sale_now'] = (today - df['dated_added']).dt.days
        df.loc[df['date_sold_valid'].notnull(), 'days_on_sale_now'] = None
        
        invalid = df['date_sold'].notnull() & df['date_sold_valid'].isnull()
        if self.debug and invalid.sum() > 0:
            self.logger.warning(f"Invalid dates: {invalid.sum()}")
        
        return df
    
    def _valid_date_sold(self, val):
        if pd.isnull(val):
            return None
        try:
            dt = pd.to_datetime(val)
            if isinstance(dt, pd.Timestamp):
                dt = dt.normalize()
            return dt if dt <= self.snapshot_date else None
        except Exception:
            return None
    
    @staticmethod
    def categorize_mileage(x) -> str:
        try:
            km = float(x)
        except Exception:
            return 'unknown'
        
        if km < 10000: return 'to_10k'
        elif km < 50000: return '10k_50k'
        elif km < 100000: return '50k_100k'
        elif km < 150000: return '100k_150k'
        elif km < 200000: return '150k_200k'
        elif km < 300000: return '200k_300k'
        else: return 'over_300k'
    
    def add_mileage_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        df['skupina_tachometru'] = df['tachometr'].apply(self.categorize_mileage)
        return df
    
    def get_group_cols(self) -> List[str]:
        return ['manufacture', 'model', 'rok_vyroby', 'fuel', 'gearbox', 'skupina_tachometru']
    
    def calc_basic_agg(self, df: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        
        agg = (df.groupby(gc)['price']
               .agg(['min', 'mean', 'max', 'count'])
               .reset_index()
               .rename(columns={'min': 'min_price', 'mean': 'avg_price', 
                              'max': 'max_price', 'count': 'total_count'}))
        
        agg['cat1_limit'] = ((agg['min_price'] + (agg['max_price'] - agg['min_price']) / 3).round(0)).astype(int)
        agg['cat2_limit'] = ((agg['min_price'] + 2 * (agg['max_price'] - agg['min_price']) / 3).round(0)).astype(int)
        
        return agg
    
    @staticmethod
    def price_cat(price: float, cat1: float, cat2: float) -> int:
        if price <= cat1: return 1
        elif price <= cat2: return 2
        else: return 3
    
    def add_price_cats(self, df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
        gc = self.get_group_cols()
        df = df.merge(agg[gc + ['cat1_limit', 'cat2_limit']], on=gc, how='left')
        df['price_cat'] = df.apply(lambda r: self.price_cat(r['price'], r['cat1_limit'], r['cat2_limit']), axis=1)
        
        if self.debug:
            self.logger.info(f"Price categories: {df['price_cat'].value_counts().to_dict()}")
        return df
