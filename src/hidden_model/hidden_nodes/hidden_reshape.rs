use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::{error::VerificationError, model::nodes::NodeProof, quantization::QSmallType, Poly};

use super::HiddenNodeOps;

pub(crate) struct HiddenReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    padded_input_shape_log: Vec<usize>,
    padded_output_shape_log: Vec<usize>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> HiddenReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(
        padded_input_shape_log: Vec<usize>,
        padded_output_shape_log: Vec<usize>,
    ) -> Self {
        Self {
            padded_input_shape_log,
            padded_output_shape_log,
            phantom: PhantomData,
        }
    }
}

impl<F, S, PCS> HiddenNodeOps<F, S, PCS> for HiddenReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        self.padded_output_shape_log.clone()
    }

    fn verify(&self, vk: &PCS::VerifierKey, proof: NodeProof) -> Result<(), VerificationError> {
        unimplemented!()
    }
}
