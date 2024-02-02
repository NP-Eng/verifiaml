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
    input_dimension_logs: Vec<usize>,
    output_dimension_logs: Vec<usize>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> NodeOps<F, S, PCS> for ReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type NodeCommitment = ();
    type InputData = QSmallType;
    type OutputData = QSmallType;
    type Proof = (); // TODO to decide

    fn log_num_units(&self) -> usize {
        return self.output_dimension_logs.iter().sum();
    }

    fn evaluate(&self, input: QArray<Self::InputData>) -> QArray<Self::OutputData> {
        // Sanity checks
        // TODO systematise
        assert!(
            zip(
                self.input_dimension_logs.iter(),
                input.shape().iter()
            ).all(|(l, d)| 1 << l == *d),
            "Received input shape does not match node input shape"
        );

        // TODO better way than cloning? Receive mutable reference?
        let mut output = input.clone();
        output.reshape(self.output_dimension_logs.iter().map(|l| 1 << l).collect());

        output
    }

    fn commit(&self) -> Self::NodeCommitment {
        // TODO assuming we want to make the reshape parameters public info,
        // no commitment is needed
        ()
    }

    fn prove(
        node_com: Self::NodeCommitment,
        input: QArray<Self::InputData>,
        input_com: PCS::Commitment,
        output: QArray<Self::OutputData>,
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
    pub(crate) fn new(input_dimension_logs: Vec<usize>, output_dimension_logs: Vec<usize>) -> Self {
        assert_eq!(
            input_dimension_logs.iter().sum::<usize>(),
            output_dimension_logs.iter().sum::<usize>(),
            "Input and output shapes have a different number of entries",
        );

        // TODO re-introduce
        // for d in input_dimensions.iter().chain(output_dimensions.iter()) {
        //     assert!(
        //         d.is_power_of_two(),
        //         "All dimensions must be powers of two"
        //     );
        // }

        Self {
            input_dimension_logs,
            output_dimension_logs,
            phantom: PhantomData,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
