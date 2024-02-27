use ark_std::log2;
use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::qarray::QTypeArray;
use crate::model::Poly;
use crate::quantization::QSmallType;

use super::{NodeOpsCommon, NodeOpsNative};

// Rectified linear unit node performing x |-> max(0, x).
pub struct ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    num_units: usize,
    log_num_units: usize,
    pub zero_point: QSmallType,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> NodeOpsNative for ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.num_units]
    }

    fn evaluate(&self, input: &QTypeArray) -> QTypeArray {
        // TODO sanity checks (cf. BMM); systematise
        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("ReLU node expects QSmallType as its QArray input type"),
        };

        QTypeArray::S(input.maximum(self.zero_point))
    }
}

// impl NodeOpsSnark
impl<F, S, PCS> NodeOpsCommon<F, S, PCS> for ReLUNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.log_num_units]
    }

    fn com_num_vars(&self) -> usize {
        0
    }
}

impl<F, S, PCS> ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub fn new(num_units: usize, zero_point: QSmallType) -> Self {
        let log_num_units = log2(num_units.next_power_of_two()) as usize;

        Self {
            num_units,
            log_num_units,
            zero_point,
            phantom: PhantomData,
        }
    }
}
