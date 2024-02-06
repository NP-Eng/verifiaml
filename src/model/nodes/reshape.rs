use std::iter::zip;
use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::QSmallType;

use super::NodeOps;

pub(crate) struct ReshapeNode<F, S, PCS> where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    padded_input_shape_logs: Vec<usize>,
    padded_output_shape_logs: Vec<usize>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> NodeOps<F, S, PCS> for ReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type NodeCommitment = ();
    type Proof = (); // TODO to decide

    fn shape(&self) -> Vec<usize> {
        self.output_shape.clone()
    }

    fn padded_shape_log(&self) -> Vec<usize> {
        self.padded_output_shape_logs.clone()
    }

    fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // Sanity checks
        // TODO systematise
        assert_eq!(
            *input.shape(),
            self.input_shape,
            "Received input shape does not match node input shape"
        );

        // TODO better way than cloning? Receive mutable reference?
        let mut output = input.clone();
        output.reshape(self.output_shape);

        output
    }

    fn commit(&self) -> Self::NodeCommitment {
        // TODO assuming we want to make the reshape parameters public info,
        // no commitment is needed
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

impl<F, S, PCS> ReshapeNode<F, S, PCS> where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_dimension: Vec<usize>, output_dimension_logs: Vec<usize>) -> Self {
        assert_eq!(
            input_dimension_logs.iter().sum::<usize>(),
            output_dimension_logs.iter().sum::<usize>(),
            "Input and output shapes have a different number of entries",
        );

        Self {
            input_dimension_logs,
            output_dimension_logs,
            phantom: PhantomData,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
