use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use ark_std::log2;
use ark_std::rand::RngCore;

use crate::model::qarray::QArray;
use crate::model::Poly;
use crate::quantization::{
    requantise_fc, FCQInfo, QInfo, QLargeType, QScaleType, QSmallType, RoundingScheme,
};
use crate::{Commitment, CommitmentState};

use super::{NodeCommitment, NodeCommitmentState, NodeOps, NodeOpsSNARK};

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

pub(crate) struct LooseFCNodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    weight_com: PCS::Commitment,
    bias_com: PCS::Commitment,
}

impl<F, S, PCS> Commitment for LooseFCNodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
}

pub(crate) struct LooseFCNodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    weight_com_state: PCS::CommitmentState,
    bias_com_state: PCS::CommitmentState,
}

impl<F, S, PCS> CommitmentState for LooseFCNodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
}

pub(crate) struct LooseFCNodeProof {
    // this will be the sumcheck proof
}

impl<F, S, PCS> NodeOps for LooseFCNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.dims.1]
    }

    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_dims_log.1]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_dims_log.0 + self.padded_dims_log.1
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
}

impl<F, S, PCS> NodeOpsSNARK<F, S, PCS> for LooseFCNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        let num_vars_weights = self.padded_dims_log.0 + self.padded_dims_log.1;
        let padded_weights_f: Vec<F> = self
            .padded_stretched_weights
            .iter()
            .map(|w| F::from(*w))
            .collect();

        let weight_poly = LabeledPolynomial::new(
            "weight_poly".to_string(),
            Poly::from_evaluations_vec(num_vars_weights, padded_weights_f),
            None,
            None, // TODO decide!
        );

        let padded_bias_f: Vec<F> = self.padded_bias.iter().map(|b| F::from(*b)).collect();

        let bias_poly = LabeledPolynomial::new(
            "bias_poly".to_string(),
            Poly::from_evaluations_vec(self.padded_dims_log.1, padded_bias_f),
            Some(self.padded_dims_log.1), // TODO or Some(1)!!
            None,                         // TODO decide!
        );

        let coms = PCS::commit(&ck, vec![&weight_poly, &bias_poly], rng).unwrap();

        (
            NodeCommitment::LooseFC(LooseFCNodeCommitment {
                weight_com: coms.0[0].commitment().clone(),
                bias_com: coms.0[1].commitment().clone(),
            }),
            NodeCommitmentState::LooseFC(LooseFCNodeCommitmentState {
                weight_com_state: coms.1[0].clone(),
                bias_com_state: coms.1[1].clone(),
            }),
        )
    }

    fn prove(
        &self,
        node_com: NodeCommitment<F, S, PCS>,
        input: QArray<QSmallType>,
        input_com: PCS::Commitment,
        output: QArray<QSmallType>,
        output_com: PCS::Commitment,
    ) -> super::NodeProof {
        todo!()
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
