use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

use ark_std::rand::RngCore;
use hcs_common::{
    LabeledPoly, Node, NodeCommitment, NodeCommitmentState, NodeOpsCommon, NodeProof, Poly,
    QTypeArray,
};

mod model;
mod nodes;

#[cfg(test)]
#[cfg(feature = "test-types")]
mod compatibility;

pub use model::ProveModel;

pub trait NodeOpsProve<F, S, PCS>: NodeOpsCommon<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QTypeArray) -> QTypeArray;

    /// Produce a node output proof
    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        node_com_state: &NodeCommitmentState<F, S, PCS>,
        input: &LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: &PCS::CommitmentState,
        output: &LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS>;

    /// Commit to the node parameters
    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>);
}

fn node_as_node_ops_snark<F, S, PCS>(node: &Node<F, S, PCS>) -> &dyn NodeOpsProve<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    match node {
        Node::BMM(fc) => fc,
        Node::RequantiseBMM(r) => r,
        Node::ReLU(r) => r,
        Node::Reshape(r) => r,
    }
}

impl<F, S, PCS> NodeOpsProve<F, S, PCS> for Node<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QTypeArray) -> QTypeArray {
        node_as_node_ops_snark(self).padded_evaluate(input)
    }

    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        node_com_state: &NodeCommitmentState<F, S, PCS>,
        input: &LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: &PCS::CommitmentState,
        output: &LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        node_as_node_ops_snark(self).prove(
            ck,
            s,
            node_com,
            node_com_state,
            input,
            input_com,
            input_com_state,
            output,
            output_com,
            output_com_state,
        )
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        node_as_node_ops_snark(self).commit(ck, rng)
    }
}
