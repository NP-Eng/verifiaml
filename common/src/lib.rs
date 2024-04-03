#![feature(generic_const_exprs)]

#[macro_use]
pub(crate) mod utils;

pub(crate) mod model;
pub(crate) mod quantization;

pub mod cp_snark;

pub use model::nodes::{
    bmm::{BMMNode, BMMNodeCommitment, BMMNodeCommitmentState, BMMNodeProof},
    relu::ReLUNode,
    requantise_bmm::{RequantiseBMMNode, RequantiseBMMNodeProof},
    reshape::ReshapeNode,
    Node, NodeCommitment, NodeCommitmentState, NodeOpsPadded, NodeProof,
};
pub use model::qarray::{InnerType, QArray, QTypeArray};
pub use model::{InferenceProof, Model};
pub use model::{LabeledPoly, Poly};
pub use quantization::{quantise_f32_u8_nne, requantise_fc, BMMQInfo, QInfo, RoundingScheme};

#[cfg(feature = "test-types")]
pub use utils::{pcs_types::Ligero, test_sponge::test_sponge};

#[cfg(feature = "test-types")]
pub mod compatibility;

#[cfg(feature = "test-types")]
pub use compatibility::example_models::{simple_perceptron_mnist, two_layer_perceptron_mnist};
