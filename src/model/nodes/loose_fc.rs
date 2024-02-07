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
    /// Unpadded dimensions (rows, columns)
    dims: (usize, usize),
    /// Dimensions of the 2D
    /// TODO generalise to 3D
    reshape_input_dims: (usize, usize),
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
        let shifted_input = (input - self.q_info.input_info.zero_point as QLargeType).move_values();

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
            1 << padded_dims.0,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_dims.0,
            input.len()
        );

        let input: QArray<QLargeType> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = (input - self.q_info.input_info.zero_point as QLargeType).move_values();
        
        let mut accumulators = self.bias.clone();

        // Padding the bias
        accumulators.resize(padded_dims.1, 0);

        // TODO this can be made more elegant (efficient?) using addition of QArrays after defining suitable operators

        // TODO since we have acumulators, this can be done more efficiently going row-wise to avoid re-caching the input
        for col in 0..padded_dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..padded_dims.0 {
                accumulators[col] +=
                    shifted_input[row] * (self.padded_stretched_weights[row * padded_dims.1 + col] as QLargeType)
            }
        }

        requantise_fc(&accumulators, &self.q_info, RoundingScheme::NearestTiesEven).into()
    }

    fn commit(&self) -> Self::NodeCommitment {
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

        let padded_stretched_weights = pad_stretch_weights(&weights, dims, reshape_input_dims);

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
            dims,
            reshape_input_dims,
            padded_dims_log,
            q_info,
            phantom: PhantomData,
        }
    }
}

// Stretch and pad the weight matrix to account for the interaction of padding
// and flattening
// weights: unpadded weight matrix, flattened
fn pad_stretch_weights(
    weights: &Vec<QSmallType>,
    dims: (usize, usize),
    original_dims: (usize, usize),
) -> Vec<QSmallType> {
    // No checks: this is an internal function

    let (n_weight_rows, n_weight_cols) = dims;
    let (n_original_rows, n_original_cols) = original_dims;

    let n_padded_rows = n_original_rows.next_power_of_two();
    let n_padded_cols = n_original_cols.next_power_of_two();
    let n_padded_weight_cols = n_weight_cols.next_power_of_two();

    let mut output = Vec::with_capacity(n_padded_rows * n_padded_cols * n_padded_weight_cols);

    let mut rows = weights.chunks_exact(n_weight_cols);

    let single_row_pad: Vec<QSmallType> = vec![0; n_padded_weight_cols - n_weight_cols];
    let full_row_pad: Vec<QSmallType> = vec![0; n_padded_weight_cols];

    for _ in 0..n_original_rows {
        for _ in 0..n_original_cols {
            output.extend(rows.next().unwrap());
            output.extend_from_slice(&single_row_pad);
        }

        for _ in 0..(n_padded_cols - n_original_cols) {
            output.extend_from_slice(&full_row_pad);
        }
    }

    output.extend(vec![
        0;
        n_padded_weight_cols
            * n_padded_cols
            * (n_padded_rows - n_original_rows)
    ]);

    output
}

// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension

#[cfg(test)]
mod tests {
    use crate::quantization::QSmallType;

    use super::pad_stretch_weights;

    #[test]
    fn weight_padding_test() {
        // Written out for visual comparison with the padded version
        let original_weights: Vec<QSmallType> = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
        ];

        // Dimensions of FC input before Reshape
        let original_dimensions = (5, 3);
        // Padded to 8 x 4

        // Dimensions of original weight matrix after Reshape of the input
        let dims = (15, 6);
        // Padded weight matrix: 32 x 8

        let expected: Vec<QSmallType> = vec![
            1, 2, 3, 4, 5, 6, 0, 0, 7, 8, 9, 10, 11, 12, 0, 0, 13, 14, 15, 16, 17, 18, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 19, 20, 21, 22, 23, 24, 0, 0, 25, 26, 27, 28, 29, 30, 0, 0, 31, 32,
            33, 34, 35, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 38, 39, 40, 41, 42, 0, 0, 43, 44, 45,
            46, 47, 48, 0, 0, 49, 50, 51, 52, 53, 54, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 56, 57, 58,
            59, 60, 0, 0, 61, 62, 63, 64, 65, 66, 0, 0, 67, 68, 69, 70, 71, 72, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 73, 74, 75, 76, 77, 78, 0, 0, 79, 80, 81, 82, 83, 84, 0, 0, 85, 86, 87, 88,
            89, 90, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];

        assert_eq!(
            super::pad_stretch_weights(&original_weights, dims, original_dimensions),
            expected
        );
    }
}
