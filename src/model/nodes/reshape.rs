use ark_std::log2;
use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::rand::RngCore;

use crate::model::qarray::{QArray, QTypeArray};
use crate::model::{LabeledPoly, Poly};
use crate::quantization::QSmallType;

use super::{NodeCommitment, NodeOps, NodeOpsSNARK, NodeProof};

pub(crate) struct ReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    padded_input_shape_log: Vec<usize>,
    padded_output_shape_log: Vec<usize>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> NodeOps for ReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn shape(&self) -> Vec<usize> {
        self.output_shape.clone()
    }

    fn evaluate(&self, input: &QTypeArray) -> QTypeArray {
        // Sanity checks
        // TODO systematise

        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("Reshape node expects QSmallType as its QArray input type"),
        };

        assert_eq!(
            *input.shape(),
            self.input_shape,
            "Received input shape does not match node input shape"
        );

        let mut output = input.clone();
        output.reshape(self.output_shape.clone());

        QTypeArray::S(output)
    }
}

impl<F, S, PCS> NodeOpsSNARK<F, S, PCS> for ReshapeNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        self.padded_output_shape_log.clone()
    }

    fn com_num_vars(&self) -> usize {
        0
    }

    // TODO I think this might be broken due to the failure of commutativity
    // between product and and nearest-geq-power-of-two
    fn padded_evaluate(&self, input: &QTypeArray) -> QTypeArray {
        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("Reshape node expects QSmallType as its QArray input type"),
        };

        let padded_input_shape: Vec<usize> = self
            .padded_input_shape_log
            .iter()
            .map(|x| (1 << x) as usize)
            .collect();

        let padded_output_shape: Vec<usize> = self
            .padded_output_shape_log
            .iter()
            .map(|x| (1 << x) as usize)
            .collect();

        // Sanity checks
        // TODO systematise
        assert_eq!(
            *input.shape(),
            padded_input_shape,
            "Received padded input shape does not match node's padded input shape"
        );

        let mut unpadded_input = input.compact_resize(self.input_shape.clone(), 0);

        // TODO only handles 2-to-1 reshapes, I think
        unpadded_input.reshape(self.output_shape.clone());
        let padded_output = unpadded_input.compact_resize(padded_output_shape, 0);

        QTypeArray::S(padded_output)
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (
        super::NodeCommitment<F, S, PCS>,
        super::NodeCommitmentState<F, S, PCS>,
    ) {
        todo!()
    }

    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input: LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: PCS::CommitmentState,
        output: LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        unimplemented!()
    }
}

impl<F, S, PCS> ReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        assert_eq!(
            input_shape.iter().product::<usize>(),
            output_shape.iter().product::<usize>(),
            "Input and output shapes have a different number of entries",
        );

        // TODO does this break the invariant that the product of I and O coincides?
        let padded_input_shape_log = input_shape
            .iter()
            .map(|x| log2(x.next_power_of_two()) as usize)
            .collect();
        let padded_output_shape_log = output_shape
            .iter()
            .map(|x| log2(x.next_power_of_two()) as usize)
            .collect();

        Self {
            input_shape,
            output_shape,
            padded_input_shape_log,
            padded_output_shape_log,
            phantom: PhantomData,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
