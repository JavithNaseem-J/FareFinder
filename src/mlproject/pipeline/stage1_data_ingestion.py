from mlproject.config.config import ConfigurationManager
from mlproject.components.component import DataIngestion
from mlproject import logger
import os

class DataIngestionPipeline:
    def __init__(self):
        pass
    def main(self):
        logger.info("Starting data ingestion pipeline")
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        logger.info(f"Data ingestion config: {data_ingestion_config}")
        data_ingestion = DataIngestion(config=data_ingestion_config)
        logger.info("Downloading file...")
        data_ingestion.download_file()
        logger.info(f"Checking if data.zip exists: {os.path.exists(data_ingestion_config.local_data_file)}")
        logger.info("Extracting zip file...")
        data_ingestion.extract_zip_file()
        logger.info(f"Checking if flight-fare-data.csv exists: {os.path.exists(os.path.join(data_ingestion_config.unzip_dir, 'flight-fare-data.csv'))}")
        logger.info("Data ingestion completed successfully")