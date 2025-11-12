
from config_manager import ConfigManager, setup_logging
from merged_table_manager import MergedTableManager

def main():
    setup_logging()
    config = ConfigManager()
    manager = MergedTableManager(config.get_engine(), config.schema)
    manager.process_all_snapshots()

if __name__ == '__main__':
    main()
