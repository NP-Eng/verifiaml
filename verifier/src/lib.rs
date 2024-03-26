use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};

use hcs_common::{InnerType, Node, NodeCommitment, NodeProof, Poly};

mod model;
mod nodes;

pub use model::VerifyModel;

pub trait NodeOpsVerify<F, S, PCS>
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

impl<F, S, PCS, ST, LT> NodeOpsVerify<F, S, PCS> for Node<ST, LT, F>
where
    F: PrimeField + Absorb + From<ST>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
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

fn node_as_node_ops_snark<F, S, PCS, ST, LT>(
    node: &Node<ST, LT, F>,
) -> &dyn NodeOpsVerify<F, S, PCS>
where
    F: PrimeField + Absorb + From<ST>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType,
{
    match node {
        Node::BMM(fc) => fc,
        Node::RequantiseBMM(r) => r,
        Node::ReLU(r) => r,
        Node::Reshape(r) => r,
    }
}
