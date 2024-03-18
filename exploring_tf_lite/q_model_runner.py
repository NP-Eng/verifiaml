from typing import Any, Tuple

import sys
sys.path.append('../exploring_tf_lite')

from example_models.q_model_wrapper import QModelWrapper
from example_models.two_layer_perceptron import QTwoLayerPerceptron

def get_model(model_name: str) -> Tuple[QModelWrapper, type]:
    """
    Returns a quantized model wrapper together with the type of the input data.
    """
    if model_name == 'two_layer_perceptron':
        return (QTwoLayerPerceptron(), int)
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