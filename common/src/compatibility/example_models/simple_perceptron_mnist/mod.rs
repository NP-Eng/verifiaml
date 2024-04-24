use crate::{
    model::nodes::{
        requantize_bmm_ref::RequantizeBMMRefNode, requantize_bmm_single::RequantizeBMMSingleNode,
    }, quantization::BMMRequantizationStrategy, BMMNode, Model, NodeOpsNative, Poly, RequantizeBMMFloatNode, ReshapeNode, Tensor
};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

pub mod parameters;
use parameters::*;

pub const INPUT_DIMS: &[usize] = &[28, 28];
pub const OUTPUT_DIM: usize = 10;

// This is the cleaner way to format a fixed string with various data due to
// the time at which Rust expands macros
macro_rules! PATH {
    () => {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/compatibility/example_models/simple_perceptron_mnist/parameters/{}"
        )
    };
}

// TODO this is incorrect now that we have switched to logs
pub fn build_simple_perceptron_mnist<F, S, PCS>(
    req_strategy: BMMRequantizationStrategy,
) -> Model<i8>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let w_array: Tensor<i8> = Tensor::read(&format!(PATH!(), "weights.json"));
    let b_array: Tensor<i32> = Tensor::read(&format!(PATH!(), "bias.json"));

    let bmm: BMMNode<i8> = BMMNode::new(w_array, b_array, Z_I);

    let boxed_req_bmm: Box<dyn NodeOpsNative<i8>> = match req_strategy {
        BMMRequantizationStrategy::Floating => Box::new(
            RequantizeBMMFloatNode::new(OUTPUT_DIM, S_I, Z_I, S_W, Z_W, S_O, Z_O),
        ),
        BMMRequantizationStrategy::Reference => Box::new(
           RequantizeBMMRefNode::new(OUTPUT_DIM, S_I, S_W, S_O, Z_O),
        ),
        BMMRequantizationStrategy::SingleRound => Box::new(
        RequantizeBMMSingleNode::new(OUTPUT_DIM, S_I, S_W, S_O, Z_O),
        ),
    };

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![Box::new(reshape), Box::new(bmm), boxed_req_bmm],
    )
}
