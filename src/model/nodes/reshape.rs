use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::QSmallType;

use super::NodeOps;

pub(crate) struct ReshapeNode<F, S, PCS> {
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

    fn num_units(&self) -> usize {
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

        // the two checks below should be performed upon construction
        if self.output_dimensions.len() != 1 {
            panic!("Only reshaping to 1 dimension is currently supported");
        }
        if self.input_dimensions.iter().product::<usize>() != self.output_dimensions[0] {
            panic!("Input and output dimensions do not match");
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

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
