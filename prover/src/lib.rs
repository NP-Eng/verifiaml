use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::rand::RngCore;

use hcs_common::{
    LabeledPoly, Node, NodeCommitment, NodeCommitmentState, NodeProof, Numeric, Poly,
};

mod model;
mod nodes;
#[macro_use]
mod util;

pub use model::ProveModel;

/// SNARK-specific operations that each node must implement.
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

impl<F, S, PCS, I, O> NodeOpsProve<F, S, PCS, I, O> for Node<I, O>
where
    F: PrimeField + Absorb + From<I> + From<O> + From<O>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    I: Numeric + TryFrom<O>,
    O: Numeric + From<I>,
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
        node_operation!(
            self,
            prove,
            ck,
            s,
            node_com,
            node_com_state,
            input,
            input_com,
            input_com_state,
            output,
            output_com,
            output_com_state
        )
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        node_operation!(self, commit, ck, rng)
    }
}
