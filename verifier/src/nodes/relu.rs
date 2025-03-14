use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use hcs_common::{NodeCommitment, NodeProof, Poly, ReLUNode};

use crate::NodeOpsVerify;

impl<F, S, PCS, ST> NodeOpsVerify<F, S, PCS> for ReLUNode<ST>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn verify(
        &self,
        _vk: &PCS::VerifierKey,
        _sponge: &mut S,
        _node_com: &NodeCommitment<F, S, PCS>,
        _input_com: &LabeledCommitment<PCS::Commitment>,
        _output_com: &LabeledCommitment<PCS::Commitment>,
        _proof: NodeProof<F, S, PCS>,
    ) -> bool {
        true
    }
}
