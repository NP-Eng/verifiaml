use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::{
    requantise_fc, FCQInfo, QInfo, QLargeType, QScaleType, QSmallType, RoundingScheme,
};

use super::NodeOps;

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Temporary Fully Connected node which also compactifies the non-compact
/// input received from a Reshape node.
pub(crate) struct LooseFCNode<F, S, PCS> {
    /// The row-major flattened unpadded vector of weights
    weights: Vec<QSmallType>,
    /// The padded and stretched weight vector
    padded_stretched_weights: Vec<QSmallType>,
    /// The unpadded vector of biases
    bias: Vec<QLargeType>,
    /// The padded bias vector. No stretching is necessary.
    padded_bias: Vec<QLargeType>,
    /// Unpadded dimensions (rows, columns)
    dims: (usize, usize),
    /// The logarithm of the padded dimensions (rows, columns)
    padded_dims_log: (usize, usize),
    /// Quantisation info used for both result computation and requantisation
    q_info: FCQInfo,

    phantom: PhantomData<(F, S, PCS)>,
}

pub(crate) struct LooseFCCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    weight_com: PCS::Commitment,
    bias_com: PCS::Commitment,
}

pub(crate) struct LooseFCProof {
    // this will be the sumcheck proof
}

impl<F, S, PCS> NodeOps<F, S, PCS> for LooseFCNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type NodeCommitment = LooseFCCommitment<F, S, PCS>;
    type Proof = LooseFCProof;

    fn shape(&self) -> Vec<usize> {
        vec![self.dims.1]
    }

    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_dims_log.1]
    }

    fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: Fully connected node expected a 1-dimensional input array"
        );
        assert_eq!(
            self.dims.0,
            input.len(),
            "Length mismatch: Fully connected node expected input with {} elements, got {} elements instead",
            self.dims.0,
            input.len()
        );

        let input: QArray<QLargeType> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = input - self.q_info.input_info.zero_point as QLargeType;

        let mut accumulators = self.bias.clone();

        // TODO this can be made more elegant (efficient?) using addition of QArrays after defining suitable operators

        // TODO since we have acumulators, this can be done more efficiently going row-wise to avoid re-caching the input
        for col in 0..self.dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..self.dims.0 {
                accumulators[col] +=
                    shifted_input[row] * (self.weights[row * self.dims.1 + col] as QLargeType)
            }
        }

        requantise_fc(&accumulators, &self.q_info, RoundingScheme::NearestTiesEven).into()
    }

    // This function naively computes entries which are known to be zero. It is
    // meant to exactly mirror the proof-system multiplication proved by the
    // sumcheck argument. Requantisation and shifting are also applied to these
    // trivial entries, as the proof system does.
    fn padded_evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        let padded_dims = (1 << self.padded_dims_log.0, 1 << self.padded_dims_log.1);

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: Fully connected node expected a 1-dimensional input array"
        );

        assert_eq!(
            padded_dims.0,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_dims.0,
            input.len()
        );

        let input: QArray<QLargeType> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = input - self.q_info.input_info.zero_point as QLargeType;

        let mut accumulators = self.padded_bias.clone();

        // TODO this can be made more elegant (efficient?) using addition of QArrays after defining suitable operators

        // TODO since we have acumulators, this can be done more efficiently going row-wise to avoid re-caching the input
        for col in 0..padded_dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..padded_dims.0 {
                accumulators[col] += shifted_input[row]
                    * (self.padded_stretched_weights[row * padded_dims.1 + col] as QLargeType)
            }
        }

        requantise_fc(&accumulators, &self.q_info, RoundingScheme::NearestTiesEven).into()
    }

    fn commit(&self, ck: PCS::CommitterKey) -> Self::NodeCommitment {
        unimplemented!()
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

    fn verify(com: Self::NodeCommitment, proof: Self::Proof) -> bool {
        unimplemented!()
    }
}

impl<F, S, PCS> LooseFCNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(
        weights: Vec<QSmallType>,
        bias: Vec<QLargeType>,
        dims: (usize, usize),
        reshape_input_dims: (usize, usize),
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

        // TODO reshape-related sanity checks

        let padded_dims_log: (usize, usize) = (
            log2(dims.0.next_power_of_two()) as usize,
            log2(dims.1.next_power_of_two()) as usize,
        );

        let array_dims = vec![reshape_input_dims.0, reshape_input_dims.1, dims.1];
        let padded_array_dims = array_dims
            .clone()
            .iter()
            .map(|x| x.next_power_of_two())
            .collect();

        // Padding and stretching the weights
        let weight_array = QArray::new(weights.clone(), array_dims);

        let padded_stretched_weights = weight_array
            .compact_resize(padded_array_dims, 0)
            .move_values();

        // Padding the bias
        let mut padded_bias = bias.clone();
        padded_bias.resize(dims.1.next_power_of_two(), 0);

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
            padded_stretched_weights,
            bias,
            padded_bias,
            dims,
            padded_dims_log,
            q_info,
            phantom: PhantomData,
        }
    }
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
