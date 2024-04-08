use crate::{BMMNode, Model, Node, Poly, QArray, RequantiseBMMNode};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

pub mod parameters;
use parameters::*;

pub const INPUT_DIM: usize = 784;
pub const OUTPUT_DIM: usize = 10;

// This is the cleaner way to format a fixed string with various data due to
// the time at which Rust expands macros
macro_rules! PATH {
    () => {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/compatibility/example_models/fully_connected_layer/parameters/{}"
        )
    };
}

pub fn build_fully_connected_layer_mnist<F, S, PCS>() -> Model<i8, i32>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let w_array: QArray<i8> = QArray::read(&format!(PATH!(), "weights.json"));
    let b_array: QArray<i32> = QArray::read(&format!(PATH!(), "bias.json"));

    let bmm: BMMNode<i8, i32> = BMMNode::new(w_array, b_array, Z_I);

    let req_bmm: RequantiseBMMNode<i8> =
        RequantiseBMMNode::new(OUTPUT_DIM, S_I, Z_I, S_W, Z_W, S_O, Z_O);

    Model::new(
        vec![INPUT_DIM],
        vec![Node::BMM(bmm), Node::RequantiseBMM(req_bmm)],
    )
}
