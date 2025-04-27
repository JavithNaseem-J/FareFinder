from mlproject.config.config import ConfigurationManager
from mlproject.components.component import DataCleaning

class DataCleaningPipeline:
    def __init__(self):
        pass

    def main(self):
        config_manager = ConfigurationManager()
        data_cleaning_config = config_manager.get_data_cleaning_config()
        data_cleaning = DataCleaning(config=data_cleaning_config)
        data_cleaning.clean_data()