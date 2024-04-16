use hcs_common::{
    test_sponge,
    two_layer_perceptron_mnist::{build_two_layer_perceptron_mnist, parameters::*, OUTPUT_DIM},
    BMMRequantizationStrategy, Ligero,
};

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;

#[path = "../common/lib.rs"]
mod common;
use common::*;

macro_rules! PATH {
    () => {
        "prover/examples/two_layer_perceptron_mnist/{}"
    };
}

fn main() {
    let two_layer_perceptron = build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        BMMRequantizationStrategy::Floating,
    );

    // Right now this can't be QInfo because the latter is always a pair
    // (f32, i8), which indeed matches in-model quantisation, but not
    // model-input quantisation (which is done outside the model). In the
    // future we can consider whether to make QInfo generic on the types and
    // make this into a QInfo
    let qinfo = (S_INPUT, Z_INPUT);

    println!("\nEXAMPLE: two-layer perceptron");
    println!("-----------------------------");

    // We need to construct the sponge outside the common/lib functions to keep
    // the latter generic on the sponge type
    let sponge: PoseidonSponge<Fr> = test_sponge();

    let output_shape = vec![OUTPUT_DIM];

    prove_inference::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &two_layer_perceptron,
        qinfo,
        sponge.clone(),
        output_shape.clone(),
    );

    verify_inference::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &two_layer_perceptron,
        qinfo,
        sponge,
        output_shape,
    );
}
