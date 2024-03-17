use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::{fmt::Debug, rand::RngCore};

use hcs_common::{
    InnerType, LabeledPoly, Node, NodeCommitment, NodeCommitmentState, NodeOpsCommon, NodeProof,
    Poly, QTypeArray,
};

mod model;
mod nodes;

pub use model::ProveModel;

pub trait NodeOpsProve<F, S, PCS, ST, LT>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QTypeArray<ST, LT>) -> QTypeArray<ST, LT>;

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

fn node_as_node_ops_snark<F, S, PCS, ST, LT>(
    node: &Node<ST, LT>,
) -> &dyn NodeOpsProve<F, S, PCS, ST, LT>
where
    F: PrimeField + Absorb + From<ST> + From<LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
    match node {
        Node::BMM(fc) => fc,
        Node::RequantiseBMM(r) => r,
        Node::ReLU(r) => r,
        Node::Reshape(r) => r,
    }
}

impl<F, S, PCS, ST, LT> NodeOpsProve<F, S, PCS, ST, LT> for Node<ST, LT>
where
    F: PrimeField + Absorb + From<ST> + From<LT> + From<LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QTypeArray<ST, LT>) -> QTypeArray<ST, LT> {
        node_as_node_ops_snark::<F, S, PCS, ST, LT>(self).padded_evaluate(input)
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
