use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::{requantise_fc, FCQInfo, QInfo, QLargeType, QScaleType, QSmallType, RoundingScheme};

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

    fn log_num_units(&self) -> usize {
        log2(self.dims.1) as usize
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
            RoundingScheme::NaiveNearestEven,
        ).into()
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

impl<F, S, PCS> FCNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(
        weights: Vec<QSmallType>,
        bias: Vec<QLargeType>,
        dims: (usize, usize),
        s_i: QScaleType,
        z_i: QSmallType,
        s_w: QScaleType,
        z_w: QSmallType,
        s_o: QScaleType,
        z_o: QSmallType,
    ) -> Self {
        assert_eq!(
            weights.len(),
            dims.0 * dims.1,
            "Weights vector length does not match the product of the dimensions"
        );

        assert_eq!(
            bias.len(),
            dims.1,
            "Bias vector length does not match the number of columns"
        );

        // TODO re-introduce
        // assert!(
        //     dims.0.is_power_of_two() && dims.1.is_power_of_two(),
        //     "Dimensions must be powers of two",
        // ); 

        let q_info = FCQInfo {
            input_info: QInfo {
                scale: s_i,
                zero_point: z_i,
            },
            weight_info: QInfo {
                scale: s_w,
                zero_point: z_w,
            },
            output_info: QInfo {
                scale: s_o,
                zero_point: z_o,
            },
        };

        Self {
            weights,
            bias,
            dims,
            q_info,
            phantom: PhantomData,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
