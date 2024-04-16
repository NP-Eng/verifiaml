from math import prod
import tensorflow as tf
import numpy as np
from .q_model_wrapper import QModelWrapper
from typing import Any, Dict, List

class QFullyConnectedLayer(QModelWrapper):

    MODEL = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, name='dense_1')
    ])

    FILENAME = '../exploring_tf_lite/models/fully_connected_layer.tflite'

    def __init__(self, args: Dict[str, str]) -> None:
        (overwrite_cache, resize_factor) = args.get("overwrite_cache", ""), args.get("resize_factor", "1")
        (overwrite_cache, resize_factor) = overwrite_cache == "True", int(resize_factor)

        super().__init__(
            QFullyConnectedLayer.FILENAME,
            model = QFullyConnectedLayer.MODEL,
            dataset = QFullyConnectedLayer.__resize_and_reshape_mnist_data(resize_factor),
            overwrite_cache=overwrite_cache
        )
        
    def get_input(self, input_index: int) -> List:
        return self.dataset[1][0][input_index]
    
    def get_output(self, input_index: int) -> List:
        return super().get_output(self.dataset[1][0][input_index])[0]
        
    def get_model_parameters(self) -> Dict[str, Any]:
        return {
            "weights" : self.quantized_model.get_tensor(2),
            "bias": self.quantized_model.get_tensor(1),
        }
    
    @staticmethod
    def __resize_and_reshape_mnist_data(resize_factor) -> np.ndarray:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        resize_and_reshape_image = lambda image: QFullyConnectedLayer.__reshape(QFullyConnectedLayer.__resize(image, resize_factor))
        return (
            (np.array(list(map(resize_and_reshape_image, x_train))), y_train),
            (np.array(list(map(resize_and_reshape_image, x_test))), y_test)
        )

    @staticmethod
    def __resize(image: tf.Tensor, resize_factor: int) -> tf.Tensor:
        new_shape = list(map(lambda x: x*resize_factor, image.shape))
        return tf.image.resize(image[..., tf.newaxis], new_shape, method="nearest")[..., 0]
    
    @staticmethod
    def __reshape(image: tf.Tensor) -> tf.Tensor:
        return tf.reshape(image, (prod(image.shape), 1))[..., 0]
