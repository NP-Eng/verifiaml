from abc import abstractmethod
from collections import OrderedDict
import json
from math import prod
import os
import tensorflow as tf
import numpy as np
from typing import Any, Dict, List, Tuple, Union

class QModelWrapper:
    """
    A wrapper for quantized models that provides a consistent interface for accessing the model's input, output, and intermediate layers.
    """

    def __init__(self, filename: str, model: tf.keras.Model = None, dataset: Tuple = None, overwrite_cache: bool = False) -> None:
        if filename is None or (model is None or dataset is None):
            raise ValueError("Either a filename or a model and dataset must be provided.")
        
        # Rescale the dataset
        rescale = lambda x: x.astype(np.float32) / 255.0
        (x_train, y_train), (x_test, y_test) = dataset
        self.dataset = (rescale(x_train), y_train), (rescale(x_test), y_test)
        (x_train, y_train), (x_test, y_test) = self.dataset

        # Train and quantize the model if it doesn't exist or if the cached file should be overwritten
        if not os.path.exists(filename) or overwrite_cache:
            with open(filename, 'wb') as f:
                f.write(
                    QModelWrapper.__quantize(
                        QModelWrapper.__train(
                            model, x_train, y_train
                        ),x_train
                    )
                )
                
        # Load the quantized model into an interpreter
        # Note: In order to access the intermediate tensors, the interpreter must be created with 
        # the flag experimental_preserve_all_tensors=True.
        # Warning: this flag does affect the output and therefore compatibility with the reference
        # Rust inference engine
        self.quantized_model = tf.lite.Interpreter(model_path=filename)
        
        
    @staticmethod
    def __train(model: tf.keras.Model, x_train: np.ndarray, y_train: np.ndarray) -> tf.keras.Model:

        # Training configuration
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train the model
        model.fit(x_train, y_train, epochs=5)
        
        return model
    
    @staticmethod
    def __quantize(model: tf.keras.Model, x_train: np.ndarray, representative_data_samples: int = 1000) -> bytes:

        # Quantize the model
        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(representative_data_samples):
                yield [input_value]
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        return converter.convert()
    
    def __quantize_input(self, input_tensor: np.ndarray) -> np.ndarray:
        input_scale, input_zero_point = self.quantized_model.get_input_details()[0]["quantization"]
        scaled_input = np.rint(input_tensor / input_scale) + input_zero_point
        clipped_input = np.clip(scaled_input, 0, 255).astype(np.uint8)
        return np.expand_dims(clipped_input, axis=0)
    
    def get_output(self, input_data: np.ndarray) -> np.ndarray:
        # Quantize the input
        input_tensor = self.__quantize_input(input_data)

        # Allocate the tensors
        self.quantized_model.allocate_tensors()

        # Set the input tensor
        self.quantized_model.set_tensor(
            self.quantized_model.get_input_details()[0]['index'], 
            input_tensor
        )

        # Run the model
        self.quantized_model.invoke()

        # Get the output tensor
        return self.quantized_model.get_tensor(
            self.quantized_model.get_output_details()[0]['index']
        )
    
    def save_params_as_qarray(self, path: str) -> None:
        print(f"Saving quantized model parameters to {path}")
        for key, value in self.get_model_parameters().items():
            with open(path + f'/{key}.json', 'w') as f:
                json.dump(QModelWrapper.as_qarray(value), f)

    @staticmethod
    def as_qarray(param: np.ndarray) -> Dict[str, Any]:
        flattened_param = QModelWrapper.multi_flatten(param.tolist())
        qarray_shape = list(reversed(param.shape))
        cumulative_dims = [prod(qarray_shape[i+1:]) for i in range(len(qarray_shape))]

        return OrderedDict([("f", flattened_param), ("s", qarray_shape), ("c", cumulative_dims)])
    
    @staticmethod
    def multi_flatten(x: Any) -> List[Union[int, float]]:
        if isinstance(x, list):
            return [y for z in x for y in QModelWrapper.multi_flatten(z)]
        return [x]

    @abstractmethod
    def get_model_parameters(self) -> Dict[str, Any]:
        pass
