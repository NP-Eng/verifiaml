use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

pub mod parameters;
use parameters::*;

use crate::{BMMNode, Model, Node, Poly, QArray, ReLUNode, RequantiseBMMNode, ReshapeNode};

pub const INPUT_DIMS: &[usize] = &[28, 28];
pub const INTER_DIM: usize = 28;
pub const OUTPUT_DIM: usize = 10;

// This is the cleaner way to format a fixed string with various data due to
// the time at which Rust expands macros
macro_rules! PATH {
    () => {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/compatibility/example_models/two_layer_perceptron_mnist/parameters/{}"
        )
    };
}

pub fn build_two_layer_perceptron_mnist<F, S, PCS>() -> Model<i8, i32>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let w1_array: QArray<i8> = QArray::read(&format!(PATH!(), "weights_1.json"));
    let b1_array: QArray<i32> = QArray::read(&format!(PATH!(), "bias_1.json"));
    let w2_array: QArray<i8> = QArray::read(&format!(PATH!(), "weights_2.json"));
    let b2_array: QArray<i32> = QArray::read(&format!(PATH!(), "bias_2.json"));

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
