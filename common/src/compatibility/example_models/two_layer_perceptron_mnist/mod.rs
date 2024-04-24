use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

pub mod parameters;
use parameters::*;

use crate::{
    model::nodes::{
        requantize_bmm_ref::RequantizeBMMRefNode, requantize_bmm_single::RequantizeBMMSingleNode,
    },
    quantization::BMMRequantizationStrategy,
    BMMNode, Model, NodeOpsNative, Poly, ReLUNode, RequantizeBMMFloatNode, ReshapeNode, Tensor,
};

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

pub fn build_two_layer_perceptron_mnist<F, S, PCS>(
    req_strategy: BMMRequantizationStrategy,
) -> Model<i8>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let w1_array: Tensor<i8> = Tensor::read(&format!(PATH!(), "weights_1.json"));
    let b1_array: Tensor<i32> = Tensor::read(&format!(PATH!(), "bias_1.json"));
    let w2_array: Tensor<i8> = Tensor::read(&format!(PATH!(), "weights_2.json"));
    let b2_array: Tensor<i32> = Tensor::read(&format!(PATH!(), "bias_2.json"));

    let bmm_1: BMMNode<i8> = BMMNode::new(w1_array, b1_array, Z_1_I);

    let boxed_req_bmm_1: Box<dyn NodeOpsNative<i8>> = match req_strategy {
        BMMRequantizationStrategy::Floating => Box::new(RequantizeBMMFloatNode::new(
            INTER_DIM, S_1_I, Z_1_I, S_1_W, Z_1_W, S_1_O, Z_1_O,
        )),
        BMMRequantizationStrategy::Reference => Box::new(RequantizeBMMRefNode::new(
            INTER_DIM, S_1_I, S_1_W, S_1_O, Z_1_O,
        )),
        BMMRequantizationStrategy::SingleRound => Box::new(RequantizeBMMSingleNode::new(
            INTER_DIM, S_1_I, S_1_W, S_1_O, Z_1_O,
        )),
    };

    let relu: ReLUNode<i8> = ReLUNode::new(28, Z_1_O);

    let bmm_2: BMMNode<i8> = BMMNode::new(w2_array, b2_array, Z_2_I);

    let boxed_req_bmm_2: Box<dyn NodeOpsNative<i8>> = match req_strategy {
        BMMRequantizationStrategy::Floating => Box::new(RequantizeBMMFloatNode::new(
            OUTPUT_DIM, S_2_I, Z_2_I, S_2_W, Z_2_W, S_2_O, Z_2_O,
        )),
        BMMRequantizationStrategy::Reference => Box::new(RequantizeBMMRefNode::new(
            OUTPUT_DIM, S_2_I, S_2_W, S_2_O, Z_2_O,
        )),
        BMMRequantizationStrategy::SingleRound => Box::new(RequantizeBMMSingleNode::new(
            OUTPUT_DIM, S_2_I, S_2_W, S_2_O, Z_2_O,
        )),
    };

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![
            Box::new(reshape),
            Box::new(bmm_1),
            boxed_req_bmm_1,
            Box::new(relu),
            Box::new(bmm_2),
            boxed_req_bmm_2,
        ],
    )
}
