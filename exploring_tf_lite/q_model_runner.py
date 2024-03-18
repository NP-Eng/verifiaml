from typing import Any, Tuple

import sys
sys.path.append('../exploring_tf_lite')

from example_models.q_model_wrapper import QModelWrapper
from example_models.q_two_layer_perceptron import QTwoLayerPerceptron
from example_models.q_simple_perceptron import QSimplePerceptron

def get_model(model_name: str) -> Tuple[QModelWrapper, type]:
    """
    Returns a quantized model wrapper together with the type of the input data.
    """
    if model_name == 'two_layer_perceptron_mnist':
        return (QTwoLayerPerceptron(), int)
    elif model_name == 'simple_perceptron_mnist':
        return (QSimplePerceptron(), int)
    else:
        raise ValueError(f"Model '{model_name}' not found.")
    
def get_model_parameters(model_name: str) -> Tuple:
    """
    Returns the model parameters as a tuple.
    """
    return get_model(model_name)[0].get_model_parameters()

def get_model_output(model_name: str, input_data: Any) -> Any:
    """
    Returns the model output on a given input.
    """
    model, input_type = get_model(model_name)
    assert isinstance(input_data, input_type), f"Input data must be of type {input_type}."
    return model.get_output(input_data)

def get_model_input(model_name: str, input_index: int) -> Any:
    """
    Returns the input data at a given index.
    """
    model, _ = get_model(model_name)
    return model.get_input(input_index)