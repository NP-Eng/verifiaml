use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::rand::RngCore;

use hcs_common::{
    LabeledPoly, NodeCommitment, NodeCommitmentState, NodeProof, Poly, ReshapeNode, SmallNIO,
};

use crate::NodeOpsProve;

impl<F, S, PCS, ST> NodeOpsProve<F, S, PCS, ST> for ReshapeNode
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
{
    fn prove(
        &self,
        _ck: &PCS::CommitterKey,
        _s: &mut S,
        _node_com: &NodeCommitment<F, S, PCS>,
        _node_com_state: &NodeCommitmentState<F, S, PCS>,
        _input: &LabeledPoly<F>,
        _input_com: &LabeledCommitment<PCS::Commitment>,
        _input_com_state: &PCS::CommitmentState,
        _output: &LabeledPoly<F>,
        _output_com: &LabeledCommitment<PCS::Commitment>,
        _output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        NodeProof::Reshape(())
    }

    fn commit(
        &self,
        _ck: &PCS::CommitterKey,
        _rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        (
            NodeCommitment::Reshape(()),
            NodeCommitmentState::Reshape(()),
        )
    }
}
