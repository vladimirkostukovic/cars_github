
import re
import json
import logging
from datetime import datetime
from typing import List

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


class SnapshotManager:
    
    PATTERN = 'sauto_%'
    REGEX = r'sauto_(\d{2})(\d{2})(\d{4})'
    
    def __init__(self, engine: Engine, schema: str = 'acars'):
        self.engine = engine
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def list_snapshots(self) -> List[str]:
        sql = text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = :s AND table_name LIKE :p"
        )
        rows = self.engine.execute(sql, s=self.schema, p=self.PATTERN).fetchall()
        
        snapshots = []
        for row in rows:
            name = row[0]
            m = re.search(self.REGEX, name)
            if m:
                date = datetime(int(m[3]), int(m[2]), int(m[1]))
                snapshots.append((name, date))
            else:
                self.logger.warning(f"Skipping table without date: {name}")
        
        snapshots.sort(key=lambda x: x[1])
        return [s[0] for s in snapshots]
    
    def load_snapshot(self, name: str) -> pd.DataFrame:
        df = pd.read_sql(text(f'SELECT * FROM {self.schema}."{name}"'), self.engine)
        self.logger.info(f"Loaded {name}: {len(df)} records")
        return df
    
    def get_snapshot_date(self, name: str) -> datetime:
        return datetime.strptime(name[-8:], '%d%m%Y')
    
    def serialize_json_columns(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        if columns is None:
            columns = [
                'category_data', 'fuel_data', 'gearbox_data', 'locality_data',
                'manufacturer_data', 'model_data', 'premise_data', 'user_data'
            ]
        
        for col in columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) 
                    else (None if pd.isnull(x) else x)
                )
        return df
