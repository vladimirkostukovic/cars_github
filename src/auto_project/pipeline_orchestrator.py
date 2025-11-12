
import logging
from datetime import datetime
from typing import Optional

from config_manager import ConfigManager, setup_logging
from merged_table_manager import MergedTableManager
from data_transformer import DataTransformer
from price_history_tracker import PriceHistoryTracker
from analytics_engine_extended import AnalyticsEngineExtended


class PipelineOrchestrator:
    
    def __init__(self, config_path: Optional[str] = None, debug: bool = True):
        setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.debug = debug
        
        self.config = ConfigManager(config_path)
        self.engine = self.config.get_engine()
        
        self.merged_mgr = MergedTableManager(self.engine, self.config.schema)
        self.transformer = DataTransformer(self.engine, self.config.schema)
        self.price_tracker = PriceHistoryTracker(self.engine, self.config.schema)
        self.analytics = AnalyticsEngineExtended(self.engine, self.config.schema, debug=self.debug)
    
    def step1_merge(self):
        self.logger.info("="*80)
        self.logger.info("STEP 1: Processing snapshots")
        self.logger.info("="*80)
        
        start = datetime.now()
        self.merged_mgr.process_all_snapshots()
        elapsed = (datetime.now() - start).total_seconds()
        
        self.logger.info(f"Step 1 complete in {elapsed:.1f}s")
    
    def step2_transform(self):
        self.logger.info("="*80)
        self.logger.info("STEP 2: Transforming data")
        self.logger.info("="*80)
        
        start = datetime.now()
        self.transformer.run()
        elapsed = (datetime.now() - start).total_seconds()
        
        self.logger.info(f"Step 2 complete in {elapsed:.1f}s")
    
    def step3_prices(self):
        self.logger.info("="*80)
        self.logger.info("STEP 3: Tracking prices")
        self.logger.info("="*80)
        
        start = datetime.now()
        self.price_tracker.track_last_pair()
        elapsed = (datetime.now() - start).total_seconds()
        
        self.logger.info(f"Step 3 complete in {elapsed:.1f}s")
    
    def step4_analytics(self):
        self.logger.info("="*80)
        self.logger.info("STEP 4: Running analytics")
        self.logger.info("="*80)
        
        start = datetime.now()
        self.analytics.run_full()
        elapsed = (datetime.now() - start).total_seconds()
        
        self.logger.info(f"Step 4 complete in {elapsed:.1f}s")
    
    def run_full(self, skip: Optional[list] = None):
        if skip is None:
            skip = []
        
        self.logger.info("🚀 STARTING FULL PIPELINE")
        self.logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start = datetime.now()
        
        try:
            if 1 not in skip:
                self.step1_merge()
            else:
                self.logger.info("Step 1 skipped")
            
            if 2 not in skip:
                self.step2_transform()
            else:
                self.logger.info("Step 2 skipped")
            
            if 3 not in skip:
                self.step3_prices()
            else:
                self.logger.info("Step 3 skipped")
            
            if 4 not in skip:
                self.step4_analytics()
            else:
                self.logger.info("Step 4 skipped")
            
            elapsed = (datetime.now() - start).total_seconds()
            self.logger.info("="*80)
            self.logger.info("✅ PIPELINE COMPLETE")
            self.logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ PIPELINE FAILED: {e}", exc_info=True)
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline orchestrator')
    parser.add_argument('--skip', type=int, nargs='+', help='Steps to skip (1-4)')
    parser.add_argument('--step', type=int, choices=[1,2,3,4], help='Run only one step')
    parser.add_argument('--no-debug', action='store_true', help='Disable debug mode')
    
    args = parser.parse_args()
    
    orch = PipelineOrchestrator(debug=not args.no_debug)
    
    if args.step:
        if args.step == 1:
            orch.step1_merge()
        elif args.step == 2:
            orch.step2_transform()
        elif args.step == 3:
            orch.step3_prices()
        elif args.step == 4:
            orch.step4_analytics()
    else:
        orch.run_full(skip=args.skip or [])


if __name__ == '__main__':
    main()
