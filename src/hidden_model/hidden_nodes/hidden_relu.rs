use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::{error::VerificationError, model::nodes::NodeProof, quantization::QSmallType, Poly};

use super::HiddenNodeOps;

pub(crate) struct HiddenReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    log_num_units: usize,
    zero_point: QSmallType,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> HiddenReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(log_num_units: usize, zero_point: QSmallType) -> Self {
        Self {
            log_num_units,
            zero_point,
            phantom: PhantomData,
        }
    }
}

impl<F, S, PCS> HiddenNodeOps<F, S, PCS> for HiddenReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        unimplemented!()
    }

    fn verify(&self, vk: &PCS::VerifierKey, proof: NodeProof) -> Result<(), VerificationError> {
        unimplemented!()
    }
}
