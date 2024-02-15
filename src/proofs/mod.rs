use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::{model::nodes::NodeProof, qarray::QArray, quantization::QSmallType, Poly};

pub(crate) struct InferenceProof<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    // Model output tensors
    pub(crate) openings: Vec<QArray<QSmallType>>,

    // Proofs of evaluation of each of the model's nodes
    pub(crate) node_proofs: Vec<NodeProof>,

    // Proofs of opening of each of the model's outputs
    pub(crate) opening_proofs: Vec<PCS::Proof>,
}
