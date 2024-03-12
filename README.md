# VerifiaML

## Build guide

We rely on the `round_ties_even` feature, which was merged into nightly 1.77. This should be stabilized in ~March. Till then, we use `nightly-2024-01-31`.

## Testing & Examples

To run the tests, use `cargo test`. 

To run the examples, use:
```
cargo run --example <example_name> --features "test-types"
```

where `<example_name>` is one of `simple_perceptron_mnist/two_layer_perceptron_mnist`.