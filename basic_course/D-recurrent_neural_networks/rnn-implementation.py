from iml.config import config
import pandas as pd
import tensorflow.keras as k
import numpy as np
from iml.source.training_engines import RecurrentNeuralNetworkTrainingEngine


def main():
    # Preprocess the data
    # Load data
    raw_data = pd.read_csv(config.data.raw_data)
    input_dataset = raw_data[config.training.input_columns_reduced]
    output_dataset = raw_data[config.training.output_columns_reduced]

    recurrent_neural_network = RecurrentNeuralNetworkTrainingEngine(input_dataset=input_dataset, output_dataset=output_dataset)
    recurrent_neural_network.run()
    validation = recurrent_neural_network.plot_validation()


if __name__ == '__main__':
    main()