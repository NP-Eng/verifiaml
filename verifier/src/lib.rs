use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

use hcs_common::{NodeCommitment, NodeOpsPadded, NodeProof, Poly, SmallNIO};

mod model;
mod nodes;

pub use model::VerifiableModel;

pub trait NodeOpsVerify<F, S, PCS, ST>: NodeOpsPadded<ST>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
{
    fn verify(
        &self,
        vk: &PCS::VerifierKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        proof: NodeProof<F, S, PCS>,
    ) -> bool;
}
