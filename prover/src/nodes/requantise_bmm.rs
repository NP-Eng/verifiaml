use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};
use ark_std::log2;
use ark_std::rand::RngCore;
use hcs_common::{
    requantise_fc, LabeledPoly, NodeCommitment, NodeCommitmentState, NodeProof, Poly, QArray,
    QSmallType, QTypeArray, RequantiseBMMNode, RequantiseBMMNodeProof, RoundingScheme,
};

use crate::NodeOpsSNARKProve;

impl<F, S, PCS> NodeOpsSNARKProve<F, S, PCS> for RequantiseBMMNode<F, S, PCS>
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
        NodeProof::RequantiseBMM(RequantiseBMMNodeProof {})
    }
}
