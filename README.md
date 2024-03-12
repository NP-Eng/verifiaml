# VerifiaML

## Build guide

We rely on the `round_ties_even` feature, which was merged into nightly 1.77. This should be stabilized in ~March. Till then, we use `nightly-2024-01-31`.

## From `ndarray` to `QArray`

In order to save a `numpy` `ndarray` (python side) as a serialised JSON which can be directly read into a `QArray` of ours (Rust side),
- Convert the `ndarray` into an `OrderedDict` using our custom python function `tensor_to_dict` (available in several of the python notebooks)
- Pass the resulting `OrderedDict` together with the destination path to `json.dump`.

The saved JSON file can be deserialised over in Rust with `QArray::read(path: &str) -> QArray`. If instead of a single `OrderedDict`, a python list of `OrderedDict`s is passed to `json.dump`, the resulting file can be deserialised with `QArray::read_list(path: &str) -> Vec<QArray> `.

Cf. `exploring_tf_lite/training_two_layer_perceptron.ipynb` for example usage.

This can be useful when bringing over to Rust some TF Lite model parameters or inputs
