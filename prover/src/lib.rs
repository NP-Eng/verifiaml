use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::{fmt::Debug, rand::RngCore};

use hcs_common::{
    InnerType, LabeledPoly, Node, NodeCommitment, NodeCommitmentState, NodeOpsCommon, NodeProof,
    Poly, QArray, QTypeArray,
};

mod model;
mod nodes;

pub use model::ProveModel;

pub trait NodeOpsProve<F, S, PCS, I, O>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
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

pub trait NodeOpsPaddedEvaluate<I, O> {
    /// Evaluate the padded node natively
    fn padded_evaluate(&self, input: &QArray<I>) -> QArray<O>;
}

pub(crate) trait NodeOpsPaddedEvaluateWrapper<I, O>
where
    I: InnerType + TryFrom<O>,
    O: InnerType + From<I>,
{
    fn padded_evaluate(&self, input: &QTypeArray<I, O>) -> QTypeArray<I, O>;
}

impl<I, O> NodeOpsPaddedEvaluateWrapper<I, O> for Node<I, O>
where
    I: InnerType + TryFrom<O>,
    O: InnerType + From<I>,
{
    fn padded_evaluate(&self, input: &QTypeArray<I, O>) -> QTypeArray<I, O> {
        match (self, input) {
            (Node::BMM(fc), QTypeArray::S(input)) => QTypeArray::L(fc.padded_evaluate(input)),
            (Node::RequantiseBMM(r), QTypeArray::L(input)) => {
                QTypeArray::S(r.padded_evaluate(input))
            }
            (Node::ReLU(r), QTypeArray::S(input)) => QTypeArray::S(r.padded_evaluate(input)),
            (Node::Reshape(r), QTypeArray::S(input)) => QTypeArray::S(r.padded_evaluate(input)),
            _ => panic!("Invalid input type for node"),
        }
    }
}

impl<F, S, PCS, I, O> NodeOpsProve<F, S, PCS, I, O> for Node<I, O>
where
    F: PrimeField + Absorb + From<I> + From<O> + From<O>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    I: InnerType + TryFrom<O>,
    O: InnerType + From<I>,
{
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
        match self {
            Node::BMM(fc) => fc.prove(
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
            ),
            Node::RequantiseBMM(r) => r.prove(
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
            ),
            Node::ReLU(r) => r.prove(
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
            ),
            Node::Reshape(r) => NodeOpsProve::<_, _, _, I, _>::prove(
                r,
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
            ),
        }
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        match self {
            Node::BMM(fc) => fc.commit(ck, rng),
            Node::RequantiseBMM(r) => r.commit(ck, rng),
            Node::ReLU(r) => r.commit(ck, rng),
            Node::Reshape(r) => NodeOpsProve::<_, _, _, I, _>::commit(r, ck, rng),
        }
    }
}
