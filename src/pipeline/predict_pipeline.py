"""
This module contains the PredictPipeline class, which is responsible for making predictions using a trained machine learning model.

Classes:
    PredictPipeline: A class that encapsulates the logic for loading a trained model, preprocessing input data, making predictions, and postprocessing the results.

Methods:
    __init__(self, model_path): Initializes the PredictPipeline with the path to the trained model.
    load_model(self): Loads the trained model from the specified file path.
    preprocess_input(self, input_data): Preprocesses the input data before making predictions.
    predict(self, input_data): Makes predictions using the preprocessed input data.
    postprocess_output(self, predictions): Postprocesses the predictions before returning them.
"""
