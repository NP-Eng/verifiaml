use hcs_common::{
    test_sponge, BMMNode, Ligero, Model, Node, Poly, QArray, RequantiseBMMNode, ReshapeNode,
};

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

mod parameters;
use parameters::*;

#[path = "../common/lib.rs"]
mod common;
use common::*;

const INPUT_DIMS: &[usize] = &[28, 28];
const OUTPUT_DIM: usize = 10;

// This is the cleaner way to format a fixed string with various data due to
// the time at which Rust expands macros
macro_rules! PATH {
    () => {
        "prover/examples/simple_perceptron_mnist/{}"
    };
}

// TODO this is incorrect now that we have switched to logs
fn build_simple_perceptron_mnist<F, S, PCS>() -> Model<i8, i32, F>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let w_array: QArray<i8> = QArray::read(&format!(PATH!(), "parameters/weights.json"));
    let b_array: QArray<i32> = QArray::read(&format!(PATH!(), "parameters/bias.json"));

    let bmm: BMMNode<i8, i32, F> = BMMNode::new(w_array, b_array, Z_I);

    let req_bmm: RequantiseBMMNode<i8> =
        RequantiseBMMNode::new(OUTPUT_DIM, S_I, Z_I, S_W, Z_W, S_O, Z_O);

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![
            Node::Reshape(reshape),
            Node::BMM(bmm),
            Node::RequantiseBMM(req_bmm),
        ],
    )
}

fn main() {
    let simple_perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    // Right now this can't be QInfo because the latter is always a pair
    // (f32, i8), which indeed matches in-model quantisation, but not
    // model-input quantisation (which is done outside the model). In the
    // future we can consider whether to make QInfo generic on the types and
    // make this into a QInfo
    let qinfo = (S_INPUT, Z_INPUT);

    println!("\nEXAMPLE: simple perceptron");
    println!("--------------------------");

    run_unpadded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
    );

    run_padded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
    );

    // MNIST test samples with index
    // 6393, 1894, 5978, 6120, 817, 3843, 7626, 9272, 498, 4622
    multi_run_unpadded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/10_test_inputs.json"),
        &format!(PATH!(), "data/10_test_outputs.json"),
        &simple_perceptron,
        qinfo,
    );

    multi_run_padded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/10_test_inputs.json"),
        &format!(PATH!(), "data/10_test_outputs.json"),
        &simple_perceptron,
        qinfo,
    );

    // We need to construct the sponge outside the common/lib functions to keep
    // the latter generic on the sponge type
    let sponge: PoseidonSponge<Fr> = test_sponge();

    let output_shape = vec![OUTPUT_DIM];

    prove_inference::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
        sponge.clone(),
        output_shape.clone(),
    );

    verify_inference::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &simple_perceptron,
        qinfo,
        sponge,
        output_shape,
    );
}
