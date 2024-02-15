use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::{
    error::VerificationError,
    model::nodes::{loose_fc::LooseFCNodeCommitment, NodeProof},
    quantization::FCQInfo,
    Poly,
};

use super::HiddenNodeOps;

pub(crate) struct HiddenLooseFCNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    padded_dims_log: (usize, usize),
    q_info: FCQInfo,
    com: LooseFCNodeCommitment<F, S, PCS>,
}

impl<F, S, PCS> HiddenNodeOps<F, S, PCS> for HiddenLooseFCNode<F, S, PCS>
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
