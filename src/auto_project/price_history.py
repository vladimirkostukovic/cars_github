
from config_manager import ConfigManager, setup_logging
from price_history_tracker import PriceHistoryTracker

def main():
    setup_logging()
    config = ConfigManager()
    tracker = PriceHistoryTracker(config.get_engine(), config.schema)
    tracker.track_last_pair()

if __name__ == '__main__':
    main()
