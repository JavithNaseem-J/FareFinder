from mlproject.config.config import ConfigurationManager
from mlproject.components.component import ModelEvaluation


class ModelEvaluationPipeline():
    def __init__(self):
        pass

    def main(self):
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            metrics = model_evaluation.evaluate()