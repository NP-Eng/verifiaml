use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::{requantise_fc, FCQInfo, QLargeType, QSmallType, RoundingScheme};

use super::NodeOps;

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Start with 2D matrices, and Mat-by-vector multiplication only
pub(crate) struct FCNode<F, S, PCS> {
    /// The row-major flattened vector of weights
    weights: Vec<QSmallType>,
    /// The vector of biases
    bias: Vec<QLargeType>,
    /// Dimensions (rows, columns)
    dims: (usize, usize),
    /// Quantisation info used for both result computation and requantisation
    q_info: FCQInfo,

    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> NodeOps<F, S, PCS> for FCNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type Commitment = PCS::Commitment;

    // this will be the sumcheck proof
    type Proof = PCS::Proof;

    fn num_units(&self) -> usize {
        self.dims.1
    }

    fn evaluate(&self, input: QArray) -> QArray {
        if input.check_dimensions().unwrap().len() != 1 {
            panic!("FC node expects a 1-dimensional array");
        }

        let input = input.values()[0][0].clone();

        assert_eq!(input.len(), self.dims.0);

        let shifted_input: Vec<_> = input
            .iter()
            .map(|x| (x - self.q_info.input_info.zero_point) as QLargeType)
            .collect();

        let mut accumulators = self.bias.clone();

        for col in 0..self.dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..self.dims.0 {
                accumulators[col] +=
                    shifted_input[row] * self.weights[row * self.dims.1 + col] as QLargeType;
            }
        }

        requantise_fc(
            &accumulators,
            &self.q_info,
            RoundingScheme::NaiveNearestAwayFromZero,
        ).into()
    }

    /// Evaluate the layer on the given input natively.
    fn commit(&self) -> PCS::Commitment {
        unimplemented!()
    }

    /// Prove that the layer was executed correctly on the given input.
    fn prove(com: PCS::Commitment, input: Vec<F>) -> PCS::Proof {
        unimplemented!()
    }

    /// Check that the layer transition was executed correctly.
    fn check(com: PCS::Commitment, proof: PCS::Proof) -> bool {
        unimplemented!()
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
