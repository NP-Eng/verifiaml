use hcs_common::{
    test_sponge, BMMNode, Ligero, Model, Node, Poly, QArray, ReLUNode, RequantiseBMMNode,
    ReshapeNode,
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
const INTER_DIM: usize = 28;
const OUTPUT_DIM: usize = 10;

// This is the cleaner way to format a fixed string with various data due to
// the time at which Rust expands macros
macro_rules! PATH {
    () => {
        "prover/examples/two_layer_perceptron_mnist/{}"
    };
}

fn build_two_layer_perceptron_mnist<F, S, PCS>() -> Model<i8, i32>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let w1_array: QArray<i8> = QArray::read(&format!(PATH!(), "parameters/weights_1.json"));
    let b1_array: QArray<i32> = QArray::read(&format!(PATH!(), "parameters/bias_1.json"));
    let w2_array: QArray<i8> = QArray::read(&format!(PATH!(), "parameters/weights_2.json"));
    let b2_array: QArray<i32> = QArray::read(&format!(PATH!(), "parameters/bias_2.json"));

    let bmm_1: BMMNode<i8, i32> = BMMNode::new(w1_array, b1_array, Z_1_I);

    let req_bmm_1: RequantiseBMMNode<i8> =
        RequantiseBMMNode::new(INTER_DIM, S_1_I, Z_1_I, S_1_W, Z_1_W, S_1_O, Z_1_O);

    let relu: ReLUNode<i8> = ReLUNode::new(28, Z_1_O);

    let bmm_2: BMMNode<i8, i32> = BMMNode::new(w2_array, b2_array, Z_2_I);

    let req_bmm_2: RequantiseBMMNode<i8> =
        RequantiseBMMNode::new(OUTPUT_DIM, S_2_I, Z_2_I, S_2_W, Z_2_W, S_2_O, Z_2_O);

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![
            Node::Reshape(reshape),
            Node::BMM(bmm_1),
            Node::RequantiseBMM(req_bmm_1),
            Node::ReLU(relu),
            Node::BMM(bmm_2),
            Node::RequantiseBMM(req_bmm_2),
        ],
    )
}

fn main() {
    let two_layer_perceptron =
        build_two_layer_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    // Right now this can't be QInfo because the latter is always a pair
    // (f32, i8), which indeed matches in-model quantisation, but not
    // model-input quantisation (which is done outside the model). In the
    // future we can consider whether to make QInfo generic on the types and
    // make this into a QInfo
    let qinfo = (S_INPUT, Z_INPUT);

    println!("\nEXAMPLE: two-layer perceptron");
    println!("-----------------------------");

    run_unpadded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &two_layer_perceptron,
        qinfo,
    );

    run_padded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/input_test_150.json"),
        &format!(PATH!(), "data/output_test_150.json"),
        &two_layer_perceptron,
        qinfo,
    );

    // MNIST test samples with index
    // 6393, 1894, 5978, 6120, 817, 3843, 7626, 9272, 498, 4622
    multi_run_unpadded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/10_test_inputs.json"),
        &format!(PATH!(), "data/10_test_outputs.json"),
        &two_layer_perceptron,
        qinfo,
    );

    multi_run_padded::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(
        &format!(PATH!(), "data/10_test_inputs.json"),
        &format!(PATH!(), "data/10_test_outputs.json"),
        &two_layer_perceptron,
        qinfo,
    );

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
