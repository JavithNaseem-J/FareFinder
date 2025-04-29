from mlproject.constants import *
from mlproject.utils.common import read_yaml, create_directories
from mlproject.entities.config_entity import DataIngestionConfig, DataValidationConfig,DataCleaningConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            status_file=config.status_file,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )
        return data_validation_config
    


    def get_data_cleaning_config(self) -> DataCleaningConfig:
        config = self.config.data_cleaning
        schema = self.schema.data_cleaning

        create_directories([config.root_dir])

        data_cleaning_config = DataCleaningConfig(
            root_dir=config.root_dir,
            input_data_path=config.input_data,
            cleaned_file=config.cleaned_file,
            file_status=config.file_status,
            columns_to_drop=schema.columns_to_drop,
            datetime_columns=schema.datetime_columns,
            target_column_mapping=schema.target_column_mapping
        )

        return data_cleaning_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        schema = self.schema
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            target_column=config.target_column,
            preprocessor_path=Path(config.preprocessor_path),
            label_encoder=Path(config.label_encoder),
            categorical_columns=schema.categorical_columns,
            numerical_columns=schema.numeric_columns
        )
        
        return data_transformation_config
    


    def get_model_training_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.GradientBoostingRegressor
        schema = self.schema
        random_search_params = params.random_search
        cv_params = params.cross_validation

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            target_column=schema.target_column.name,
            random_search_params=random_search_params, 
            cv_folds=cv_params.cv_folds,            
            scoring=cv_params.scoring,             
            n_jobs=cv_params.n_jobs,
            n_iter=cv_params.n_iter          
        )
        
        return model_trainer_config
    



    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.GradientBoostingRegressor
        schema = self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_raw_data=config.test_raw_data,
            model_path=config.model_path,
            all_params=params,
            metric_file_path=config.metric_file_path,
            preprocessor_path=config.preprocessor_path,
            target_column=schema.name,
        )
        return model_evaluation_config