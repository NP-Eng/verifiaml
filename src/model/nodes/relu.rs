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
    log_num_units: usize,
    phantom: PhantomData<(F, S, PCS)>,
}

pub(crate) struct ReLUProof {
    // this will be a lookup proof
}

impl<F, S, PCS> NodeOps<F, S, PCS> for ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type NodeCommitment = ();
    type Proof = ReLUProof;

    fn log_num_units(&self) -> usize {
        self.log_num_units
    }

    fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // TODO sanity checks (cf. FC); systematise
        
        // TODO Can be done more elegantly, probably
        let v: Vec<QSmallType> = input.values()
            .iter()
            .map(|x| *max(x, &(0 as QSmallType)))
            .collect();
        
        v.into()
    }

    fn commit(&self) -> Self::NodeCommitment {
        // ReLU nodes have no parameters to commit to
        ()
    }

    fn prove(
        node_com: Self::NodeCommitment,
        input: QArray<QSmallType>,
        input_com: PCS::Commitment,
        output: QArray<QSmallType>,
        output_com: PCS::Commitment,
    ) -> Self::Proof {
        unimplemented!()
    }

    fn verify(node_com: Self::NodeCommitment, proof: Self::Proof) -> bool {
        unimplemented!()
    }
}

impl<F, S, PCS> ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(log_num_units: usize) -> Self {
        Self {
            log_num_units,
            phantom: PhantomData,
        }
    }
}
