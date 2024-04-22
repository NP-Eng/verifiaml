# VerifiaML

## Testing & Examples

To run the tests, use `cargo test`. 

To run the examples, use:
```
cargo run --example <example_name> --features "test-types"
```

where `<example_name>` is one of the following:

- `simple_perceptron_mnist_inference`
- `simple_perceptron_mnist_proof`
- `two_layer_perceptron_mnist_inference`
- `two_layer_perceptron_mnist_proof`

In order to run any tests involving python code, such as compatibility tests with TF Lite, the feature `python` must be activated (which automatically enables `test-types`).

## From `ndarray` to `Tensor`

In order to save a `numpy` `ndarray` (python side) as a serialised JSON which can be directly read into a `Tensor` of ours (Rust side),
- Convert the `ndarray` into an `OrderedDict` using our custom python function `tensor_to_dict` (available in several of the python notebooks)
- Pass the resulting `OrderedDict` together with the destination path to `json.dump`.

The saved JSON file can be deserialised over in Rust with `Tensor::read(path: &str) -> Tensor`. If instead of a single `OrderedDict`, a python list of `OrderedDict`s is passed to `json.dump`, the resulting file can be deserialised with `Tensor::read_list(path: &str) -> Vec<Tensor> `.

Cf. `exploring_tf_lite/training_two_layer_perceptron.ipynb` for example usage.

This can be useful when bringing over to Rust some TF Lite model parameters or inputs.
