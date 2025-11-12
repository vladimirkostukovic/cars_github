
import re
import json
import logging
from typing import Optional

import pandas as pd
from sqlalchemy.engine import Engine


class DataTransformer:
    
    def __init__(self, engine: Engine, schema: str = 'acars'):
        self.engine = engine
        self.schema = schema
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @staticmethod
    def extract_json_field(val, key: str) -> Optional[str]:
        if pd.isnull(val):
            return None
        if isinstance(val, dict):
            return val.get(key)
        try:
            return json.loads(val).get(key)
        except Exception:
            return None
    
    @staticmethod
    def extract_equipment(name: str) -> Optional[str]:
        if not isinstance(name, str):
            return None
        
        parts = [p.strip() for p in name.split(',')]
        if len(parts) <= 1:
            return None
        
        equipment = ', '.join(parts[1:])
        
        # Remove irrelevant words
        bad = [
            r'\bÄŒR\b', r'\b1\.?maj\b', r'\bmajitel\b', r'koupeno v Är', 
            r'\bAC\b', r'\bklima\b', r'\bserv\.?kniha\b', r'\belectric\b', 
            r'\bautomat\b', r'\broadster\b', r'\bcabrio\b', r'\bavangarde\b', r'\btourer\b'
        ]
        for b in bad:
            equipment = re.sub(b, '', equipment, flags=re.IGNORECASE)
        
        equipment = re.sub(r'[ ,]+', ' ', equipment).strip()
        return equipment or None
    
    @staticmethod
    def extract_motor(name: str) -> Optional[str]:
        if not isinstance(name, str):
            return None
        
        patterns = [
            r'(\d{1,2}[.,]\d{1,2})\s*(Turbo|TSI|TFSI|CDTI|CDI|i|D)?',
            r'\b(\d{1,2}[.,]\d{1,2})\b',
            r'\b(\d{3})\s?(CDI|d|i)\b',
            r'\b(\d{1,2})\s?V\b',
        ]
        
        for pat in patterns:
            m = re.search(pat, name, re.IGNORECASE)
            if m:
                res = m.group(1).replace(',', '.')
                try:
                    if float(res) > 10:
                        continue
                except Exception:
                    pass
                return res
        return None
    
    @staticmethod
    def extract_power(name: str) -> Optional[int]:
        if not isinstance(name, str):
            return None
        
        patterns = [
            r'([0-9]{2,4})\s*[kK][wW]',
            r'([0-9]{2,4})\s*[hH][pP]',
            r'([0-9]{2,4})\s*[pP][sS]',
            r'([0-9]{2,4})\s*[kK][vV]',
        ]
        
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return int(m.group(1))
        return None
    
    def load_merged_data(self) -> pd.DataFrame:
        self.logger.info('Reading merged table...')
        df = pd.read_sql(f'SELECT * FROM {self.schema}.merged', self.engine)
        self.logger.info(f'Loaded {len(df)} rows from merged')
        
        sold = df['date_sold'].notnull().sum()
        active = df['date_sold'].isnull().sum()
        self.logger.info(f'In merged: active={active}, sold={sold}')
        
        return df
    
    def transform_to_standart(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info('Extracting motor and power...')
        df['motor'] = df['name'].apply(self.extract_motor)
        df['power'] = df['name'].apply(self.extract_power)
        
        standart = pd.DataFrame({
            'id': df['id'],
            'internal_id': df['internal_id'],
            'additional_model_name': df['additional_model_name'],
            'custom_id': df['custom_id'],
            'deal_type': df['deal_type'],
            'manuf_date': df['manufacturing_date'],
            'cebia': df['is_cebia_smart_code_url_verified'],
            'name': df['name'],
            'price': df['price'],
            'tachometr': df['tachometer'],
            'category': df['category_data'].apply(lambda x: self.extract_json_field(x, 'name')),
            'fuel': df['fuel_data'].apply(lambda x: self.extract_json_field(x, 'name')),
            'gearbox': df['gearbox_data'].apply(lambda x: self.extract_json_field(x, 'name')),
            'locality': df['locality_data'].apply(lambda x: self.extract_json_field(x, 'region')),
            'manufacture': df['manufacturer_data'].apply(lambda x: self.extract_json_field(x, 'name')),
            'model': df['model_data'].apply(lambda x: self.extract_json_field(x, 'name')),
            'body': df['category_data'].apply(lambda x: self.extract_json_field(x, 'seo_name')),
            'equipment': df['name'].apply(self.extract_equipment),
            'motor': df['motor'],
            'power': df['power'],
            'seller': df['premise_data'].apply(lambda x: self.extract_json_field(x, 'name')),
            'dated_added': df['date_added'],
            'date_sold': df['date_sold']
        })
        
        self.logger.info(f'Standart dataframe: {len(standart)} rows, {standart.shape[1]} cols')
        return standart
    
    def save_standart(self, df: pd.DataFrame):
        try:
            df_old = pd.read_sql(f'SELECT internal_id FROM {self.schema}.standart', self.engine)
            old = set(df_old['internal_id'])
            new = set(df['internal_id'])
            added = len(new - old)
            removed = len(old - new)
            self.logger.info(f'Delta: added={added}, removed={removed}')
        except Exception:
            self.logger.info('Standart not found, creating from scratch')
        
        df.to_sql('standart', self.engine, schema=self.schema, 
                 if_exists='replace', index=False, method='multi')
        self.logger.info('Standart table updated')
        
        sold = df['date_sold'].notnull().sum()
        active = df['date_sold'].isnull().sum()
        self.logger.info(f'In standart: active={active}, sold={sold}')
    
    def run(self):
        df_merged = self.load_merged_data()
        df_standart = self.transform_to_standart(df_merged)
        self.save_standart(df_standart)
