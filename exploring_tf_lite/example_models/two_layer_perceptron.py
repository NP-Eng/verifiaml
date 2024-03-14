from typing import Tuple, List

import tensorflow as tf
from .q_model_wrapper import QModelWrapper

class QTwoLayerPerceptron(QModelWrapper):
    
    MODEL = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_1'),
            tf.keras.layers.Dense(28, name='dense_1'),
            tf.keras.layers.ReLU(name='relu_1'),
            tf.keras.layers.Dense(10, name='dense_2')
        ])
    
    DATASET = tf.keras.datasets.mnist.load_data()
    
    FILENAME = '../exploring_tf_lite/models/two_layer_perceptron.tflite'

    def __init__(self) -> None:
        super().__init__(
            QTwoLayerPerceptron.FILENAME,
            model = QTwoLayerPerceptron.MODEL,
            dataset = QTwoLayerPerceptron.DATASET
        )

        # MODEL PARAMETERS: (dense_1_bias, dense_1_weights, dense_2_bias, dense_2_weights)
        self.model_params = (
            self.quantized_model.get_tensor(4),
            self.quantized_model.get_tensor(5).transpose(),
            self.quantized_model.get_tensor(2),
            self.quantized_model.get_tensor(3).transpose(),
        )
    
    def get_output(self, input_index: int) -> List:
        return super().get_output(self.dataset[0][0][input_index])[0]

    def get_intermediate_outputs(self, input_index: int) -> Tuple:
        
        # Quantize the input
        input_tensor = self.__quantize_input(self.dataset[0][0][input_index])

        # Allocate the tensors
        self.quantized_model.allocate_tensors()

        # Set the input tensor
        self.quantized_model.set_tensor(
            self.quantized_model.get_input_details()[0]['index'], 
            input_tensor
        )

        # Run the model
        self.quantized_model.invoke()

        # OUTPUTS: (input, quantized_input, dense_1_output, dense_2_output, output)
        return (
            self.dataset[0][0][input_index],
            input_tensor[0],
            self.quantized_model.get_tensor(8)[0],
            self.quantized_model.get_tensor(9)[0],
            self.quantized_model.get_tensor(10)[0]
        )
        
    def get_model_parameters(self) -> Tuple:
        return self.model_params
