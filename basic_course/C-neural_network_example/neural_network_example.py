from iml.source.training_engines import NeuralNetworkTrainingEngine
from iml.source.validation_engines import NeuralNetworkValidationEngine
import pandas as pd
from iml.config import config
import tensorflow.keras as k


class CustomNeuralNetworkTrainingEngine(NeuralNetworkTrainingEngine):
    def model(self):
        self.model = k.Sequential()


if __name__ == '__main__':
    data_df = pd.read_csv(config.general.input_path)
    input_df = data_df[config.general.input_columns]
    output_df = data_df[config.general.output_columns]

    neural_network_trainer = NeuralNetworkTrainingEngine(input_dataset=input_df,
                                                         output_dataset=output_df,
                                                         )
    neural_network_trainer.run()

    random_forest_validator = NeuralNetworkValidationEngine(neural_network_trainer)
    random_forest_validator.run()
