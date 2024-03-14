use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::{fmt::Debug, rand::RngCore};

use hcs_common::{
    requantise_fc, InnerType, LabeledPoly, NodeCommitment, NodeCommitmentState, NodeProof, Poly,
    QArray, QTypeArray, RequantiseBMMNode, RequantiseBMMNodeCommitment,
    RequantiseBMMNodeCommitmentState, RequantiseBMMNodeProof, RoundingScheme,
};

use crate::NodeOpsProve;

impl<F, S, PCS, ST, LT> NodeOpsProve<F, S, PCS, ST, LT> for RequantiseBMMNode<ST>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
    fn padded_evaluate(&self, input: &QTypeArray<ST, LT>) -> QTypeArray<ST, LT> {
        let input = match input {
            QTypeArray::L(i) => i,
            _ => panic!("RequantiseBMM node expects QLargeType as its QArray input type"),
        };

        let padded_size = 1 << self.padded_size_log;

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantiseBMM node expects a 1-dimensional input array"
        );

        assert_eq!(
            padded_size,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_size,
            input.len()
        );

        let output: QArray<ST> = requantise_fc::<ST, LT>(
            input.values(),
            &self.q_info,
            RoundingScheme::NearestTiesEven,
        )
        .into();

        QTypeArray::S(output)
    }

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
        NodeProof::RequantiseBMM(RequantiseBMMNodeProof {})
    }

    fn commit(
        &self,
        _ck: &PCS::CommitterKey,
        _rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        (
            NodeCommitment::RequantiseBMM(RequantiseBMMNodeCommitment()),
            NodeCommitmentState::RequantiseBMM(RequantiseBMMNodeCommitmentState()),
        )
    }
}
