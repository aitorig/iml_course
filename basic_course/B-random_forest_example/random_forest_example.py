from iml.source.training_engines import RandomForestRegressionTrainingEngine
from iml.source.validation_engines import RandomForestValidationEngine
import pandas as pd
from iml.config import config


if __name__ == '__main__':
    data_df = pd.read_csv(config.general.input_path)
    input_df = data_df[config.general.input_columns]
    output_df = data_df[config.general.output_columns]

    random_forest_trainer = RandomForestRegressionTrainingEngine(input_dataset=input_df,
                                                                 output_dataset=output_df,
                                                                 )
    random_forest_trainer.run()

    random_forest_validator = RandomForestValidationEngine(random_forest_trainer)
    random_forest_validator.run()
