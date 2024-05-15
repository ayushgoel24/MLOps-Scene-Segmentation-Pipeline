import os
from pymongo import MongoClient
from logger.manager import LoggerManager


class ConfigManager:
    mongo_uri = os.getenv('MONGO_URI')
    mongo_user = os.getenv('MONGO_USER')
    mongo_password = os.getenv('MONGO_PASSWORD')
    database_name = os.getenv('MONGO_DB_NAME')
    
    mongo_url = f"mongodb://{mongo_user}:{mongo_password}@{mongo_uri}"
    # mongo_url = "mongodb://mongouser:mongopassword@192.168.49.237"
    _client = MongoClient(mongo_url)
    _db = _client[database_name]
    # _db = _client["segmentation"]

    @staticmethod
    def fetch_dataset_config():
        """Retrieve the dataset configuration from MongoDB."""
        collection = ConfigManager._db['configurations']
        dataset_config = collection.find_one({'config_type': 'dataset'})
        if dataset_config:
            dataset_config.pop('_id', None)  # Optionally remove the MongoDB id field
        return dataset_config

    @staticmethod
    def fetch_dataloader_config():
        """Retrieve the dataloader configuration from MongoDB."""
        collection = ConfigManager._db['configurations']
        dataloader_config = collection.find_one({'config_type': 'dataloader'})
        if dataloader_config:
            dataloader_config.pop('_id', None)
        return dataloader_config
