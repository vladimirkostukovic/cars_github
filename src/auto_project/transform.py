
from config_manager import ConfigManager, setup_logging
from data_transformer import DataTransformer

def main():
    setup_logging()
    config = ConfigManager()
    transformer = DataTransformer(config.get_engine(), config.schema)
    transformer.run()

if __name__ == '__main__':
    main()
