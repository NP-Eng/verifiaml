use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::{fmt::Debug, rand::RngCore};

use hcs_common::{
    InnerType, LabeledPoly, NodeCommitment, NodeCommitmentState, NodeProof, Poly, QArray,
    QTypeArray, ReLUNode,
};

use crate::{NodeOpsPaddedEvaluate, NodeOpsProve};

impl<F, S, PCS, ST> NodeOpsProve<F, S, PCS, ST, ST> for ReLUNode<ST>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType,
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
        NodeProof::ReLU(())
    }

    fn commit(
        &self,
        _ck: &PCS::CommitterKey,
        _rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        (NodeCommitment::ReLU(()), NodeCommitmentState::ReLU(()))
    }
}

impl<ST> NodeOpsPaddedEvaluate<ST, ST> for ReLUNode<ST>
where
    ST: InnerType,
{
    // TODO this is the same as evaluate() for now; the two will likely differ
    // if/when we introduce input size checks
    fn padded_evaluate(&self, input: &QArray<ST>) -> QArray<ST> {
        // TODO sanity checks (cf. BMM); systematise
        input.maximum(self.zero_point)
    }
}
