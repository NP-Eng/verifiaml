use hcs_common::{
    example_models::simple_perceptron_mnist::{
        build_simple_perceptron_mnist, parameters::*, OUTPUT_DIM,
    },
    test_sponge, Ligero, Model,
};

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;

#[path = "../common/lib.rs"]
mod common;
use common::*;

macro_rules! PATH {
    () => {
        "prover/examples/simple_perceptron_mnist/{}"
    };
}

fn main() {
    let simple_perceptron: Model<Fr, PoseidonSponge<Fr>, Ligero<Fr>> =
        build_simple_perceptron_mnist();

    // Right now this can't be QInfo because the latter is always a pair
    // (f32, i8), which indeed matches in-model quantisation, but not
    // model-input quantisation (which is done outside the model). In the
    // future we can consider whether to make QInfo generic on the types and
    // make this into a QInfo
    let qinfo = (S_INPUT, Z_INPUT);

    println!("\nEXAMPLE: simple perceptron");
    println!("--------------------------");

    run_unpadded(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
    );

    run_padded(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
    );

    // MNIST test samples with index
    // 6393, 1894, 5978, 6120, 817, 3843, 7626, 9272, 498, 4622
    multi_run_unpadded(
        &format!(PATH!(), "data/10_test_inputs.json"),
        &format!(PATH!(), "data/10_test_outputs.json"),
        &simple_perceptron,
        qinfo,
    );

    multi_run_padded(
        &format!(PATH!(), "data/10_test_inputs.json"),
        &format!(PATH!(), "data/10_test_outputs.json"),
        &simple_perceptron,
        qinfo,
    );

    // We need to construct the sponge outside the common/lib functions to keep
    // the latter generic on the sponge type
    let sponge: PoseidonSponge<Fr> = test_sponge();

    let output_shape = vec![OUTPUT_DIM];

    prove_inference(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
        sponge.clone(),
        output_shape.clone(),
    );

    verify_inference(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
        sponge,
        output_shape,
    );
}
