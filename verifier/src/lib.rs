use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

use hcs_common::{Node, NodeCommitment, NodeOpsSNARK, NodeProof, Poly};

mod model;
mod nodes;

pub use model::VerifyModel;

pub trait NodeOpsSNARKVerify<F, S, PCS>: NodeOpsSNARK<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
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

impl<F, S, PCS> NodeOpsSNARKVerify<F, S, PCS> for Node<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn verify(
        &self,
        vk: &PCS::VerifierKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        proof: NodeProof<F, S, PCS>,
    ) -> bool {
        node_as_node_ops_snark(self).verify(vk, s, node_com, input_com, output_com, proof)
    }
}

fn node_as_node_ops_snark<F, S, PCS>(node: &Node<F, S, PCS>) -> &dyn NodeOpsSNARKVerify<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    match node {
        Node::BMM(fc) => fc,
        Node::RequantiseBMM(r) => r,
        Node::ReLU(r) => r,
        Node::Reshape(r) => r,
    }
}
