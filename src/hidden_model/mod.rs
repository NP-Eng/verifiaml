use ark_std::{log2, rand::RngCore};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};

use crate::{
    error::VerificationError, hidden_model::hidden_nodes::HiddenNodeOps, proofs::InferenceProof,
    Poly,
};

use self::hidden_nodes::HiddenNode;

pub(crate) mod hidden_nodes;

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all nodes to use the same PCS; this might change
// in the future
pub(crate) struct HiddenModel<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    input_shape_log: Vec<usize>,
    output_shape_log: Vec<usize>,
    nodes: Vec<HiddenNode<F, S, PCS>>,
}

impl<F, S, PCS> HiddenModel<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_shape_log: Vec<usize>, nodes: Vec<HiddenNode<F, S, PCS>>) -> Self {
        // An empty model would cause problems later on
        assert!(!nodes.is_empty(), "A model cannot have no nodes",);

        Self {
            input_shape_log,
            output_shape_log: nodes.last().unwrap().padded_shape_log(),
            nodes,
        }
    }

    pub(crate) fn get_nodes(&self) -> &Vec<HiddenNode<F, S, PCS>> {
        &self.nodes
    }

    pub(crate) fn verify_inference() {
        todo!();
    }
}
