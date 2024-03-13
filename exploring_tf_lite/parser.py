import os
from math import log2, ceil, floor

import pprint
import tensorflow as tf
import numpy as np


class TwoLayerPerceptron:

    def __init__(self) -> None:
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), name='flatten_1'),
            tf.keras.layers.Dense(28, name='dense_1'),
            tf.keras.layers.ReLU(name='relu_1'),
            tf.keras.layers.Dense(10, name='dense_2')
        ])
        
        # Rescale the dataset
        rescale = lambda x: x.astype(np.float32) / 255.0
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.dataset = (rescale(x_train), y_train), (rescale(x_test), y_test)

        # Train and quantize the model if it doesn't exist
        filename = 'exploring_tf_lite/models/two_layer_perceptron.tflite'
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                f.write(
                    TwoLayerPerceptron.__quantize(
                        TwoLayerPerceptron.__train(
                            self.model, x_train, y_train
                        ),
                        x_train
                    )
                )
        
        # Load the quantized model into an interpreter
        self.quantized_model = tf.lite.Interpreter(model_path=filename, experimental_preserve_all_tensors=True)

        # Get the weights and biases
        self.dense_1_weights = self.quantized_model.get_tensor(5)
        self.dense_1_bias = self.quantized_model.get_tensor(4)

        self.dense_2_weights = self.quantized_model.get_tensor(3)
        self.dense_2_bias = self.quantized_model.get_tensor(2)

    def get_run_transcript(self, input_index: int) -> dict:
        return {
            **{
                "dense_1_bias": self.dense_1_bias,
                "dense_1_weights": self.dense_1_weights,
                "dense_2_bias": self.dense_2_bias,
                "dense_2_weights": self.dense_2_weights,
            }, 
            **self.__get_intermediate_outputs(input_index)
        }

    def __get_intermediate_outputs(self, input_index: int) -> dict:
        
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

        # Get the output tensors
        return {
            "dense_1_output": self.quantized_model.get_tensor(8),
            "dense_2_output": self.quantized_model.get_tensor(9),
            "output": self.quantized_model.get_tensor(10)
        }

    
    @staticmethod
    def __train(model: tf.keras.Model, x_train, y_train) -> tf.keras.Model:

        # Training configuration
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        # Train the model
        model.fit(x_train, y_train, epochs=5)
        
        return model
    
    def __quantize(self, model: tf.keras.Model, x_train) -> bytes:

        # Quantize the model
        def representative_data_gen():
            for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(1000):
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
    
print(TwoLayerPerceptron().get_run_transcript(150))