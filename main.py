import argparse
from mlproject import logger
from mlproject.pipeline.stage1_data_ingestion import DataIngestionPipeline
from mlproject.pipeline.stage2_data_validation import DataValidationPipeline
from mlproject.pipeline.stage3_data_cleaning import DataCleaningPipeline
from mlproject.pipeline.stage4_data_transformation import DataTransformationPipeline
from mlproject.pipeline.stage5_model_training import ModelTrainerPipeline
from mlproject.pipeline.stage6_model_evaluation import ModelEvaluationPipeline

def run_stage(stage_name):
    logger.info(f">>>>>> Stage {stage_name} started <<<<<<")

    try:
        if stage_name == "data_ingestion":
            stage = DataIngestionPipeline()
            stage.main()

        elif stage_name == "data_validation":
            stage = DataValidationPipeline()
            stage.main()

        elif stage_name == "data_cleaning":
            stage = DataCleaningPipeline()
            stage.main()

        elif stage_name == "data_transformation":
            stage = DataTransformationPipeline()
            stage.main()

        elif stage_name == "model_training":
            stage = ModelTrainerPipeline()
            stage.main()

        elif stage_name == "model_evaluation":
            stage = ModelEvaluationPipeline()
            stage.main()

        else:
            raise ValueError(f"Unknown stage: {stage_name}")

        logger.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
    except KeyError as e:
        logger.error(f"Missing key in configuration or data: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run specific pipeline stage.")
    parser.add_argument("--stage", help="Name of the stage to run")
    args = parser.parse_args()

    if args.stage:
        run_stage(args.stage)
    else:
        stages = [
            "data_ingestion",
            "data_validation",
            "data_cleaning",
            "data_transformation",
            "model_training",
            "model_evaluation",
        ]
        for stage in stages:
            run_stage(stage)
