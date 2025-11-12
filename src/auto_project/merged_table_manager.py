
import logging
from typing import Set, Tuple
from datetime import datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from snapshot_manager import SnapshotManager


class MergedTableManager:
    
    def __init__(self, engine: Engine, schema: str = 'acars'):
        self.engine = engine
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)
        self.snapshot_manager = SnapshotManager(engine, schema)
    
    def ensure_table_exists(self, df: pd.DataFrame):
        cols = [f'"{c}" TEXT' for c in df.columns if c not in ['date_added', 'date_sold']]
        
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {self.schema}.merged (
            internal_id SERIAL PRIMARY KEY,
            {", ".join(cols)},
            date_added DATE,
            date_sold DATE
        )
        """
        with self.engine.begin() as conn:
            conn.execute(text(ddl))
        self.logger.info("Merged table checked/created")
    
    def load_state(self) -> Tuple[Set[int], Set[int]]:
        try:
            df = pd.read_sql(text(f'SELECT id, date_sold FROM {self.schema}.merged'), self.engine)
            known = set(df['id'])
            active = set(df[df['date_sold'].isnull()]['id'])
            self.logger.info(f"State loaded: known={len(known)}, active={len(active)}")
            return known, active
        except Exception as e:
            self.logger.warning(f"Could not load merged (may not exist): {e}")
            return set(), set()
    
    def process_snapshot(self, name: str, known: Set[int], active: Set[int]) -> Tuple[Set[int], Set[int], int, int, int]:
        df = self.snapshot_manager.load_snapshot(name)
        date = self.snapshot_manager.get_snapshot_date(name)
        df = self.snapshot_manager.serialize_json_columns(df)
        
        current = set(df['id'])
        new_ids = current - known
        removed = active - current
        
        # Add new listings
        new_count = self._add_new(df, new_ids, date)
        
        # Mark as sold
        sold_count = self._mark_sold(removed, date)
        
        # Reactivate
        reactivated_count = self._reactivate(current)
        
        # Update state
        known |= current
        reactivated = self._get_reactivated(current) if current else set()
        active = (active | new_ids | reactivated) - removed
        
        return known, active, new_count, sold_count, reactivated_count
    
    def _add_new(self, df: pd.DataFrame, ids: Set[int], date: datetime) -> int:
        if not ids:
            self.logger.info("No new listings")
            return 0
        
        new_df = df[df['id'].isin(ids)].copy()
        new_df['date_added'] = date
        new_df['date_sold'] = None
        
        self.ensure_table_exists(new_df)
        new_df.to_sql('merged', self.engine, schema=self.schema, 
                     if_exists='append', index=False, method='multi', chunksize=20000)
        
        self.logger.info(f"Added {len(new_df)} new listings")
        return len(new_df)
    
    def _mark_sold(self, ids: Set[int], date: datetime) -> int:
        if not ids:
            self.logger.info("No sold listings")
            return 0
        
        sql = text(
            f"UPDATE {self.schema}.merged SET date_sold = :date "
            "WHERE id = ANY(:ids) AND (date_sold IS NULL OR date_sold > :date)"
        )
        with self.engine.begin() as conn:
            conn.execute(sql, {'date': date, 'ids': list(ids)})
        
        self.logger.info(f"Marked {len(ids)} as sold")
        return len(ids)
    
    def _reactivate(self, ids: Set[int]) -> int:
        if not ids:
            return 0
        
        ids_str = ','.join(map(str, ids))
        df = pd.read_sql(
            f"SELECT id, date_sold FROM {self.schema}.merged WHERE id IN ({ids_str})",
            self.engine
        )
        
        was_sold = set(df[df['date_sold'].notnull()]['id'])
        reactivated = ids & was_sold
        
        if not reactivated:
            return 0
        
        sql = text(f"UPDATE {self.schema}.merged SET date_sold = NULL "
                  f"WHERE id = ANY(:ids) AND date_sold IS NOT NULL")
        with self.engine.begin() as conn:
            conn.execute(sql, {'ids': list(reactivated)})
        
        self.logger.info(f"Reactivated {len(reactivated)} listings")
        return len(reactivated)
    
    def _get_reactivated(self, ids: Set[int]) -> Set[int]:
        if not ids:
            return set()
        ids_str = ','.join(map(str, ids))
        df = pd.read_sql(
            f"SELECT id, date_sold FROM {self.schema}.merged WHERE id IN ({ids_str})",
            self.engine
        )
        was_sold = set(df[df['date_sold'].notnull()]['id'])
        return ids & was_sold
    
    def process_all_snapshots(self):
        snapshots = self.snapshot_manager.list_snapshots()
        
        if not snapshots:
            self.logger.warning("No snapshots found")
            return
        
        self.logger.info(f"Total snapshots: {len(snapshots)}")
        
        known, active = self.load_state()
        total_new = total_sold = total_reactivated = 0
        
        for idx, snap in enumerate(snapshots, 1):
            self.logger.info(f"[{idx}/{len(snapshots)}] Processing {snap}")
            
            known, active, new, sold, reactivated = self.process_snapshot(snap, known, active)
            
            self.logger.info(f"[{snap}] Summary: new={new}, sold={sold}, reactivated={reactivated}")
            
            total_new += new
            total_sold += sold
            total_reactivated += reactivated
        
        self.logger.info("✓ Processing complete")
        self.logger.info(f"Total: new={total_new}, sold={total_sold}, reactivated={total_reactivated}")
