use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

use hcs_common::{
    requantise_fc, LabeledPoly, NodeCommitment, NodeCommitmentState, NodeProof, Poly, QArray,
    QSmallType, QTypeArray, RequantiseBMMNode, RequantiseBMMNodeProof, RoundingScheme,
};

use crate::NodeOpsProve;

impl<F, S, PCS> NodeOpsProve<F, S, PCS> for RequantiseBMMNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_evaluate(&self, input: &QTypeArray) -> QTypeArray {
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

        let output: QArray<QSmallType> = requantise_fc(
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
}
