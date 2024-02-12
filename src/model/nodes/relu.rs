use ark_std::cmp::max;
use ark_std::log2;
use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::QSmallType;

use super::NodeOpsNative;

// Rectified linear unit node performing x |-> max(0, x).
pub(crate) struct ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    num_units: usize,
    log_num_units: usize,
    zero_point: QSmallType,
    phantom: PhantomData<(F, S, PCS)>,
}

pub(crate) struct ReLUProof {
    // this will be a lookup proof
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

    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.log_num_units]
    }

    fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // TODO sanity checks (cf. FC); systematise
        input.maximum(self.zero_point)
    }

    // TODO this is the same as evaluate() for now; the two will likely differ
    // if/when we introduce input size checks
    fn padded_evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // TODO sanity checks (cf. FC); systematise
        input.maximum(self.zero_point)
    }
}

impl<F, S, PCS> ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(num_units: usize, zero_point: QSmallType) -> Self {
        let log_num_units = log2(num_units.next_power_of_two()) as usize;

        Self {
            num_units,
            log_num_units,
            zero_point,
            phantom: PhantomData,
        }
    }
}
