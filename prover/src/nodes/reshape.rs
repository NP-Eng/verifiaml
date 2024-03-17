use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::rand::RngCore;

use hcs_common::{
    InnerType, LabeledPoly, NodeCommitment, NodeCommitmentState, NodeProof, Poly, QArray,
    QTypeArray, ReshapeNode,
};

use crate::{NodeOpsPaddedEvaluate, NodeOpsProve};

impl<ST> NodeOpsPaddedEvaluate<ST, ST> for ReshapeNode
where
    ST: InnerType,
{
    // TODO I think this might be broken due to the failure of commutativity
    // between product and and nearest-geq-power-of-two
    fn padded_evaluate(&self, input: &QArray<ST>) -> QArray<ST> {
        let padded_input_shape: Vec<usize> = self
            .padded_input_shape_log
            .iter()
            .map(|x| (1 << x) as usize)
            .collect();

        let padded_output_shape: Vec<usize> = self
            .padded_output_shape_log
            .iter()
            .map(|x| (1 << x) as usize)
            .collect();

        // Sanity checks
        // TODO systematise
        assert_eq!(
            *input.shape(),
            padded_input_shape,
            "Received padded input shape does not match node's padded input shape"
        );

        let mut unpadded_input = input.compact_resize(self.input_shape.clone(), ST::ZERO);

        // TODO only handles 2-to-1 reshapes, I think
        unpadded_input.reshape(self.output_shape.clone());
        let padded_output = unpadded_input.compact_resize(padded_output_shape, ST::ZERO);
        padded_output
    }
}

impl<F, S, PCS, ST> NodeOpsProve<F, S, PCS, ST, ST> for ReshapeNode
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
