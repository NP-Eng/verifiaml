use hcs_common::{
    simple_perceptron_mnist::{build_simple_perceptron_mnist, parameters::*},
    Ligero,
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
            "/examples/simple_perceptron_mnist/data/{}"
        )
    };
}

fn main() {
    let simple_perceptron =
        build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(false);

    // Right now this can't be QInfo because the latter is always a pair
    // (f32, i8), which indeed matches in-model quantisation, but not
    // model-input quantisation (which is done outside the model). In the
    // future we can consider whether to make QInfo generic on the types and
    // make this into a QInfo
    let qinfo = (S_INPUT, Z_INPUT);

    println!("\nEXAMPLE: simple perceptron");
    println!("--------------------------");

    run_unpadded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "input_test_150.json"),
        &format!(PATH!(), "output_test_150.json"),
        &simple_perceptron,
        qinfo,
    );

    // MNIST test samples with index
    // 6393, 1894, 5978, 6120, 817, 3843, 7626, 9272, 498, 4622
    multi_run_unpadded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "10_test_inputs.json"),
        &format!(PATH!(), "10_test_outputs.json"),
        &simple_perceptron,
        qinfo,
    );
}
