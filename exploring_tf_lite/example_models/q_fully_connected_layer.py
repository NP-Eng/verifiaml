import tensorflow as tf
import numpy as np
from .q_model_wrapper import QModelWrapper
from typing import List, Tuple

class QFullyConnectedLayer(QModelWrapper):

    MODEL = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, name='dense_1')
    ])
    
    # Flatten the image data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    DATASET = (
        (np.array([np.concatenate((x,), axis=None) for x in x_train]), y_train),
        (np.array([np.concatenate((x,), axis=None) for x in x_test]), y_test)
    )
    
    FILENAME = '../exploring_tf_lite/models/fully_connected_layer.tflite'

    def __init__(self) -> None:
        super().__init__(
            QFullyConnectedLayer.FILENAME,
            model = QFullyConnectedLayer.MODEL,
            dataset = QFullyConnectedLayer.DATASET
        )

        # MODEL PARAMETERS: (dense_1_bias, dense_1_weights)
        self.model_params = (
            self.quantized_model.get_tensor(2),
            self.quantized_model.get_tensor(1).transpose(),
        )
        
    def get_input(self, input_index: int) -> List:
        return self.dataset[1][0][input_index]
    
    def get_output(self, input_index: int) -> List:
        return super().get_output(self.dataset[1][0][input_index])[0]
        
    def get_model_parameters(self) -> Tuple:
        return self.model_params
