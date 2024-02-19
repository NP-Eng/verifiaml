use ark_std::log2;
use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::rand::RngCore;

use crate::model::qarray::{QArray, QTypeArray};
use crate::model::Poly;
use crate::quantization::QSmallType;

use super::{NodeCommitment, NodeCommitmentState, NodeOps, NodeOpsSNARK};

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

impl<F, S, PCS> NodeOps for ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.num_units]
    }

    fn evaluate(&self, input: QTypeArray) -> QTypeArray {
        // TODO sanity checks (cf. FC); systematise
        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("ReLU node expects QSmallType as its QArray input type"),
        };

        QTypeArray::S(input.maximum(self.zero_point))
    }
}

// impl NodeOpsSnark
impl<F, S, PCS> NodeOpsSNARK<F, S, PCS> for ReLUNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.log_num_units]
    }

    fn com_num_vars(&self) -> usize {
        0
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        todo!()
    }

    // TODO this is the same as evaluate() for now; the two will likely differ
    // if/when we introduce input size checks
    fn padded_evaluate(&self, input: QTypeArray) -> QTypeArray {
        // TODO sanity checks (cf. FC); systematise

        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("ReLU node expects QSmallType as its QArray input type"),
        };

        QTypeArray::S(input.maximum(self.zero_point))
    }

    fn prove(
        &self,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input: QTypeArray,
        input_com: &PCS::Commitment,
        output: QTypeArray,
        output_com: &PCS::Commitment,
    ) -> super::NodeProof {
        todo!()
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
