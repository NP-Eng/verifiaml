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
    input_dimensions: Vec<usize>,
    output_dimensions: Vec<usize>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> NodeOps<F, S, PCS> for ReshapeNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type Commitment = PCS::Commitment;

    type Proof = PCS::Proof;

    fn log_num_units(&self) -> usize {
        return self.output_dimensions.iter().product();
    }

    /// Evaluate the layer on the given input natively.
    fn evaluate(&self, input: QArray) -> QArray {
        if input.check_dimensions().unwrap() != self.input_dimensions {
            panic!(
                "Incorrect input dimensions: found {:?}, expected {:?}",
                input.check_dimensions().unwrap(),
                self.input_dimensions,
            );
        }

        if self.input_dimensions.len() == 1 {
            return input;
        }
        if self.input_dimensions.len() == 2 {
            let v: Vec<QSmallType> = input.values()[0].clone().into_iter().flatten().collect();
            return v.into();
        }
        else {
            let v: Vec<QSmallType> = input.move_values().into_iter().flatten().flatten().collect();
            return v.into();
        }
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
}

impl<F, S, PCS> ReshapeNode<F, S, PCS> where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_dimensions: Vec<usize>, output_dimensions: Vec<usize>) -> Self {

        assert_eq!(
            output_dimensions.len(),
            1,
            "Only reshaping to 1 dimension is currently supported"  
        );

        assert_eq!(
            input_dimensions.iter().product::<usize>(),
            output_dimensions[0],
            "Input and output dimensions do not match",
        );

        for d in input_dimensions.iter().chain(output_dimensions.iter()) {
            assert!(
                d.is_power_of_two(),
                "All dimensions must be powers of two"
            );
        }

        Self {
            input_dimensions,
            output_dimensions,
            phantom: PhantomData,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
