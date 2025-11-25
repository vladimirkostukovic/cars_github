import subprocess
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# ============== Config ==============
SCHEMA = 'acars'


def get_engine():
    config_path = Path(__file__).resolve().parent / 'config.json'
    with config_path.open(encoding='utf-8') as f:
        config = json.load(f)
    conn_str = (
        f"postgresql+psycopg2://{config['USER']}:{config['PWD']}"
        f"@{config['HOST']}:{config['PORT']}/{config['DB']}"
    )
    return create_engine(conn_str)


# ============== Run Stats ==============
def ensure_run_stats_table(engine):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.run_stats (
        id SERIAL PRIMARY KEY,
        run_date TIMESTAMP DEFAULT NOW(),
        duration_seconds NUMERIC,
        status TEXT,
        steps_run TEXT,

        merged_total INT,
        merged_active INT,
        merged_sold INT,

        standart_total INT,
        standart_active INT,
        standart_sold INT,

        snapshots_processed INT,
        price_changes_today INT,
        analytics_groups INT,

        errors TEXT
    )
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def collect_stats(engine) -> dict:
    stats = {}

    # Merged
    try:
        merged = pd.read_sql(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN date_sold IS NULL THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN date_sold IS NOT NULL THEN 1 ELSE 0 END) as sold
            FROM {SCHEMA}.merged
        """, engine).iloc[0]
        stats['merged_total'] = int(merged['total'])
        stats['merged_active'] = int(merged['active'])
        stats['merged_sold'] = int(merged['sold'])
    except Exception:
        pass

    # Standart
    try:
        standart = pd.read_sql(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN date_sold IS NULL THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN date_sold IS NOT NULL THEN 1 ELSE 0 END) as sold
            FROM {SCHEMA}.standart
        """, engine).iloc[0]
        stats['standart_total'] = int(standart['total'])
        stats['standart_active'] = int(standart['active'])
        stats['standart_sold'] = int(standart['sold'])
    except Exception:
        pass

    # Snapshots
    try:
        snapshots = pd.read_sql(
            f"SELECT COUNT(*) as cnt FROM {SCHEMA}.processed_snapshots",
            engine
        ).iloc[0]['cnt']
        stats['snapshots_processed'] = int(snapshots)
    except Exception:
        pass

    # Price changes today
    try:
        today = datetime.now().date()
        price_changes = pd.read_sql(f"""
            SELECT COUNT(*) as cnt 
            FROM {SCHEMA}.price_history 
            WHERE change_date = '{today}'
        """, engine).iloc[0]['cnt']
        stats['price_changes_today'] = int(price_changes)
    except Exception:
        stats['price_changes_today'] = 0

    # Analytics
    try:
        analytics = pd.read_sql(
            f"SELECT COUNT(*) as cnt FROM {SCHEMA}.agg_price_cats",
            engine
        ).iloc[0]['cnt']
        stats['analytics_groups'] = int(analytics)
    except Exception:
        pass

    return stats


def save_run_stats(engine, stats: dict):
    cols = list(stats.keys())
    placeholders = ', '.join([f':{k}' for k in cols])
    columns = ', '.join(cols)

    sql = text(f"INSERT INTO {SCHEMA}.run_stats ({columns}) VALUES ({placeholders})")

    with engine.begin() as conn:
        conn.execute(sql, stats)


# ============== Pipeline ==============
def run_script(script_name: str):
    script_path = Path(__file__).parent / script_name
    result = subprocess.run([sys.executable, str(script_path)], check=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Pipeline orchestrator')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4],
                        help='Run only specific step')
    parser.add_argument('--skip', type=int, nargs='+', default=[],
                        help='Steps to skip (1-4)')
    args = parser.parse_args()

    steps = {
        1: 'step1_merge.py',
        2: '2_step_transform.py',
        4: '3_step_analytics.py',
    }

    print("=" * 60)
    print(f"🚀 PIPELINE STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # DB connection
    engine = get_engine()
    ensure_run_stats_table(engine)

    # Track stats
    run_stats = {
        'run_date': datetime.now(),
        'status': 'running',
        'steps_run': '',
        'errors': None
    }
    steps_run = []

    start = datetime.now()

    try:
        if args.step:
            if args.step in [1]:
                run_script('step1_merge.py')
                steps_run.append('1')
            elif args.step in [2, 3]:
                run_script('step2_transform.py')
                steps_run.append('2+3')
            elif args.step == 4:
                run_script('step3_analytics.py')
                steps_run.append('4')
        else:
            if 1 not in args.skip:
                print("\n>>> Running Step 1: Merge")
                run_script('step1_merge.py')
                steps_run.append('1')
            else:
                print("Step 1 skipped")

            if 2 not in args.skip and 3 not in args.skip:
                print("\n>>> Running Step 2+3: Transform & Price Tracking")
                run_script('2_step_transform.py')
                steps_run.append('2+3')
            else:
                print("Steps 2-3 skipped")

            if 4 not in args.skip:
                print("\n>>> Running Step 4: Analytics")
                run_script('3_step_analytics.py')
                steps_run.append('4')
            else:
                print("Step 4 skipped")

        elapsed = (datetime.now() - start).total_seconds()
        run_stats['duration_seconds'] = elapsed
        run_stats['status'] = 'success'
        run_stats['steps_run'] = ','.join(steps_run)

        print("=" * 60)
        print(f"✅ PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed / 60:.1f}m)")
        print("=" * 60)

    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start).total_seconds()
        run_stats['duration_seconds'] = elapsed
        run_stats['status'] = 'failed'
        run_stats['steps_run'] = ','.join(steps_run)
        run_stats['errors'] = str(e)

        print(f"❌ PIPELINE FAILED: {e}")

    # Save stats
    try:
        current_stats = collect_stats(engine)
        run_stats.update(current_stats)
        save_run_stats(engine, run_stats)
        print(f"\n📊 Stats saved to {SCHEMA}.run_stats")
    except Exception as e:
        print(f"⚠️ Could not save stats: {e}")

    return 1 if run_stats['status'] == 'failed' else 0


if __name__ == '__main__':
    sys.exit(main())