pub(crate) mod model;
pub(crate) mod quantization;
pub(crate) mod utils;

trait Commitment {}

trait CommitmentState {}

trait Proof {}

pub use model::isolated_verification::verify_inference;
pub use model::nodes::{
    bmm::{BMMNode, BMMNodeCommitment, BMMNodeCommitmentState, BMMNodeProof},
    relu::ReLUNode,
    requantise_bmm::{RequantiseBMMNode, RequantiseBMMNodeProof},
    reshape::ReshapeNode,
    Node, NodeCommitment, NodeCommitmentState, NodeOpsSNARK, NodeProof,
};
pub use model::qarray::{QArray, QTypeArray};
pub use model::{InferenceProof, Model};
pub use model::{LabeledPoly, Poly};
pub use quantization::{
    quantise_f32_u8_nne, requantise_fc, BMMQInfo, QLargeType, QSmallType, RoundingScheme,
};

#[cfg(feature = "test-types")]
pub use utils::{pcs_types::Ligero, test_sponge::test_sponge};
