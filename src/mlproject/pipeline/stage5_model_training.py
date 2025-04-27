from mlproject.config.config import ConfigurationManager
from mlproject.components.component import ModelTrainer



class ModelTrainerPipeline():
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_training_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()