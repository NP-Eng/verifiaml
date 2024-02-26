use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use hcs_common::{NodeCommitment, NodeProof, Poly, RequantiseBMMNode};

use crate::NodeOpsSNARKVerify;

impl<F, S, PCS> NodeOpsSNARKVerify<F, S, PCS> for RequantiseBMMNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn verify(
        &self,
        vk: &PCS::VerifierKey,
        sponge: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        proof: NodeProof<F, S, PCS>,
    ) -> bool {
        true
    }
}
