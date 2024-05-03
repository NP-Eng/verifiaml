from typing import Any, Dict, List

import tensorflow as tf
from .q_model_wrapper import QModelWrapper

class QSimplePerceptron(QModelWrapper):
    
    MODEL = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(10, name='dense')
    ])
    
    DATASET = tf.keras.datasets.mnist.load_data()
    
    FILENAME = '../exploring_tf_lite/models/simple_model_quant.tflite'

    def __init__(self) -> None:
        super().__init__(
            QSimplePerceptron.FILENAME,
            model = QSimplePerceptron.MODEL,
            dataset = QSimplePerceptron.DATASET
        )

    def get_input(self, input_index: int) -> List:
        return self.dataset[1][0][input_index]
    
    def get_output(self, input_index: int) -> List:
        return super().get_output(self.dataset[1][0][input_index])[0]
        
    def get_model_parameters(self) -> Dict[str, Any]:
        return {
            "weights" : self.quantized_model.get_tensor(3),
            "bias": self.quantized_model.get_tensor(2).transpose(),
        }
        
