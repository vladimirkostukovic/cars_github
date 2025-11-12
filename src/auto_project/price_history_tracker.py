
import logging
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from snapshot_manager import SnapshotManager


class PriceHistoryTracker:
    
    MIN_CHANGE = 1000  # Min price change to track
    
    def __init__(self, engine: Engine, schema: str = 'acars'):
        self.engine = engine
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)
        self.snapshot_manager = SnapshotManager(engine, schema)
    
    def ensure_table(self):
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.schema}.price_history (
            id BIGINT,
            old_price NUMERIC,
            new_price NUMERIC,
            change_date DATE
        );
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))
        self.logger.info("Price history table checked/created")
    
    def load_snapshot_prices(self, name: str) -> pd.DataFrame:
        return pd.read_sql(text(f'SELECT id, price FROM {self.schema}."{name}"'), self.engine)
    
    def find_changes(self, df_prev: pd.DataFrame, df_last: pd.DataFrame) -> pd.DataFrame:
        merged = df_last.merge(df_prev, on='id', how='inner', suffixes=('_new', '_old'))
        
        changes = merged[
            (merged['price_new'] != merged['price_old']) &
            (abs(merged['price_new'] - merged['price_old']) >= self.MIN_CHANGE)
        ].copy()
        
        return changes
    
    def filter_existing(self, changes: pd.DataFrame, date: datetime.date) -> pd.DataFrame:
        try:
            today_hist = pd.read_sql(
                f"SELECT id, change_date FROM {self.schema}.price_history WHERE change_date = %(dt)s",
                self.engine, params={"dt": date}
            )
            mask = ~changes.set_index(['id', 'change_date']).index.isin(
                today_hist.set_index(['id', 'change_date']).index
            )
            changes = changes[mask]
        except Exception as e:
            self.logger.warning(f"Could not load history for dedup: {e}")
        return changes
    
    def save_changes(self, changes: pd.DataFrame):
        if changes.empty:
            self.logger.info("All changes already recorded")
            return
        
        changes.to_sql('price_history', self.engine, schema=self.schema,
                      if_exists='append', index=False, method='multi', chunksize=5000)
        self.logger.info(f"Saved {len(changes)} price changes")
    
    def track_last_pair(self):
        snapshots = self.snapshot_manager.list_snapshots()
        
        if len(snapshots) < 2:
            self.logger.warning("Not enough snapshots to compare")
            return
        
        self.ensure_table()
        
        prev, last = snapshots[-2], snapshots[-1]
        self.logger.info(f"Comparing {prev} and {last}")
        
        df_prev = self.load_snapshot_prices(prev)
        df_last = self.load_snapshot_prices(last)
        date_last = self.snapshot_manager.get_snapshot_date(last).date()
        
        changes = self.find_changes(df_prev, df_last)
        
        if changes.empty:
            self.logger.info("No price changes found")
            return
        
        changes['change_date'] = date_last
        changes = changes[['id', 'price_old', 'price_new', 'change_date']]
        changes = changes.rename(columns={'price_old': 'old_price', 'price_new': 'new_price'})
        
        changes = self.filter_existing(changes, date_last)
        self.save_changes(changes)
    
    def get_history(self, listing_id: int) -> pd.DataFrame:
        sql = f"SELECT * FROM {self.schema}.price_history WHERE id = :id ORDER BY change_date"
        return pd.read_sql(text(sql), self.engine, params={'id': listing_id})
