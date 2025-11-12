
from config_manager import ConfigManager, setup_logging
from analytics_engine_extended import AnalyticsEngineExtended

def main():
    setup_logging()
    config = ConfigManager()
    analytics = AnalyticsEngineExtended(config.get_engine(), config.schema, debug=True)
    analytics.run_full()

if __name__ == '__main__':
    main()
