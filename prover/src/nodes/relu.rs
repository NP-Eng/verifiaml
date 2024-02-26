use std::rc::Rc;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};
use ark_sumcheck::ml_sumcheck::{protocol::ListOfProductsOfPolynomials, MLSumcheck};
use hcs_common::{
    BMMNode, BMMNodeCommitment, BMMNodeCommitmentState, BMMNodeProof, LabeledPoly, NodeCommitment,
    NodeCommitmentState, NodeProof, Poly, QArray, QLargeType, QTypeArray, ReLUNode,
};

use crate::NodeOpsSNARKProve;

impl<F, S, PCS> NodeOpsSNARKProve<F, S, PCS> for ReLUNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    // TODO this is the same as evaluate() for now; the two will likely differ
    // if/when we introduce input size checks
    fn padded_evaluate(&self, input: &QTypeArray) -> QTypeArray {
        // TODO sanity checks (cf. BMM); systematise

        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("ReLU node expects QSmallType as its QArray input type"),
        };

        QTypeArray::S(input.maximum(self.zero_point))
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
        NodeProof::ReLU(())
    }
}
