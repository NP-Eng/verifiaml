from typing import Any, Tuple

import sys
sys.path.append('../exploring_tf_lite')

from example_models.q_model_wrapper import QModelWrapper
from example_models.q_two_layer_perceptron import QTwoLayerPerceptron
from example_models.q_simple_perceptron import QSimplePerceptron

AVAILABLE_MODELS = {
    "QTwoLayerPerceptron": (QTwoLayerPerceptron, int),
    "QSimplePerceptron": (QSimplePerceptron, int)
}

def get_model(model_name: str) -> QModelWrapper:
    """
    Returns a quantized model wrapper together with the type of the input data.
    """
    model, _ = AVAILABLE_MODELS.get(model_name, (None, None))
    assert model is not None, f"Model {model_name} is not available."
    return model()
    
def get_model_parameters(model_name: str) -> Tuple:
    """
    Returns the model parameters as a tuple.
    """
    return get_model(model_name).get_model_parameters()
