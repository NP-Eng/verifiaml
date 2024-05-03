from typing import Any, Dict, List, Tuple

import sys
sys.path.append('../exploring_tf_lite')

from example_models.q_model_wrapper import QModelWrapper
from example_models.q_two_layer_perceptron import QTwoLayerPerceptron
from example_models.q_simple_perceptron import QSimplePerceptron
from example_models.q_fully_connected_layer import QFullyConnectedLayer

AVAILABLE_MODELS = {
    "QTwoLayerPerceptron": (QTwoLayerPerceptron, int),
    "QSimplePerceptron": (QSimplePerceptron, int),
    "QFullyConnectedLayer": (QFullyConnectedLayer, int)
}

def get_model(model_name: str, args: List[Tuple[str, str]] = None) -> QModelWrapper:
    """
    Returns a quantized model wrapper together with the type of the input data.
    """
    model, _ = AVAILABLE_MODELS.get(model_name, (None, None))
    assert model is not None, f"Model {model_name} is not available."
    return model(dict(args)) if args is not None else model()
