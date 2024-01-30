use core::panic;
use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::cmp::max;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::QSmallType;

use super::NodeOps;

// Rectified linear unit node performing x |-> max(0, x).
pub(crate) struct ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    log_num_nodes: usize,
    phantom: PhantomData<(F, S, PCS)>,
}

// TODO: it will be more efficient to add size checks here
impl<F, S, PCS> NodeOps<F, S, PCS> for ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type Commitment = PCS::Commitment;

    type Proof = PCS::Proof;

    fn log_num_nodes(&self) -> usize {
        self.log_num_nodes
    }

    fn evaluate(&self, input: QArray) -> QArray {
        if input.check_dimensions().unwrap().len() != 1 {
            panic!("ReLU node expects a 1-dimensional array");
        }
        
        let v: Vec<QSmallType> = input.values()[0][0]
            .iter()
            .map(|x| *max(x, &(0 as QSmallType)))
            .collect();
        
        v.into()
    }

    fn commit(&self) -> PCS::Commitment {
        unimplemented!()
    }

    fn prove(com: PCS::Commitment, input: Vec<F>) -> PCS::Proof {
        unimplemented!()
    }

    fn check(com: PCS::Commitment, proof: PCS::Proof) -> bool {
        unimplemented!()
    }

    fn num_nodes(&self) -> usize {
        todo!()
    }
}
