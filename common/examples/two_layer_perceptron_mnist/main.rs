use hcs_common::{
    two_layer_perceptron_mnist::{build_two_layer_perceptron_mnist, parameters::*},
    Ligero, Model,
};

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;

#[path = "../common/lib.rs"]
mod common;
use common::*;

macro_rules! PATH {
    () => {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/two_layer_perceptron_mnist/data/{}"
        )
    };
}

fn main() {
    let two_layer_perceptron: Model<Fr, PoseidonSponge<Fr>, Ligero<Fr>> =
        build_two_layer_perceptron_mnist();

    // Right now this can't be QInfo because the latter is always a pair
    // (f32, i8), which indeed matches in-model quantisation, but not
    // model-input quantisation (which is done outside the model). In the
    // future we can consider whether to make QInfo generic on the types and
    // make this into a QInfo
    let qinfo = (S_INPUT, Z_INPUT);

    println!("\nEXAMPLE: two-layer perceptron");
    println!("-----------------------------");

    run_unpadded(
        &format!(PATH!(), "input_test_150.json"),
        &format!(PATH!(), "output_test_150.json"),
        &two_layer_perceptron,
        qinfo,
    );

    // MNIST test samples with index
    // 6393, 1894, 5978, 6120, 817, 3843, 7626, 9272, 498, 4622
    multi_run_unpadded(
        &format!(PATH!(), "10_test_inputs.json"),
        &format!(PATH!(), "10_test_outputs.json"),
        &two_layer_perceptron,
        qinfo,
    );
}
