use ark_std::cmp::max;
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_poly_commit::PolynomialCommitment;

use crate::quantization::QSmallType;
use crate::model::Poly;

// Rectified linear unit node performing x |-> max(0, x).
pub(crate) struct ReLUNode<F, S, PCS> where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    log_num_nodes: usize,
}

// TODO: it will be more efficient to add size checks here
impl<F, S, PCS> ReLUNode<F, S, PCS> where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{

    pub(crate) fn log_num_nodes(&self) -> usize {
        self.log_num_nodes
    }

    pub(crate) fn evaluate(&self, input: Vec<QSmallType>) -> Vec<QSmallType> {
        input.map(|x| max(x, 0)).collect()
    }

    pub(crate) fn commit(&self) -> PCS::Commitment {
        unimplemented!()
    }

    pub(crate) fn prove(com: PCS::Commitment, input: Vec<F>) -> PCS::Proof {
        unimplemented!()
    }

    pub(crate) fn check(com: PCS::Commitment, proof: PCS::Proof) -> bool {
        unimplemented!()
    }
}