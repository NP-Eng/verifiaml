use hcs_common::{
    simple_perceptron_mnist::{build_simple_perceptron_mnist, parameters::*, OUTPUT_DIM},
    test_sponge, BMMRequantizationStrategy, Ligero,
};

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;

#[path = "../common/lib.rs"]
mod common;
use common::*;
use hcs_prover::{as_provable_model, as_verifiable_model};

macro_rules! PATH {
    () => {
        "prover/examples/simple_perceptron_mnist/{}"
    };
}

fn main() {
    let simple_perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        BMMRequantizationStrategy::Floating,
    );

    let (provable_model, verifiable_model) = (
        as_provable_model(&simple_perceptron),
        as_verifiable_model(&simple_perceptron),
    );

    // Right now this can't be QInfo because the latter is always a pair
    // (f32, i8), which indeed matches in-model quantisation, but not
    // model-input quantisation (which is done outside the model). In the
    // future we can consider whether to make QInfo generic on the types and
    // make this into a QInfo
    let qinfo = (S_INPUT, Z_INPUT);

    println!("\nEXAMPLE: simple perceptron");
    println!("--------------------------");

    // We need to construct the sponge outside the common/lib functions to keep
    // the latter generic on the sponge type
    let sponge: PoseidonSponge<Fr> = test_sponge();

    let output_shape = vec![OUTPUT_DIM];

    prove_inference::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &provable_model,
        qinfo,
        sponge.clone(),
        output_shape.clone(),
    );

    verify_inference::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &provable_model,
        &verifiable_model,
        qinfo,
        sponge,
        output_shape,
    );
}
