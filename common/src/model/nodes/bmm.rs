use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::log2;

use ark_sumcheck::ml_sumcheck::Proof;

use crate::model::qarray::{QArray, QTypeArray};
use crate::model::Poly;
use crate::quantization::{QLargeType, QSmallType};
use crate::{Commitment, CommitmentState};

use super::{NodeOpsCommon, NodeOpsNative};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Start with 2D matrices, and Mat-by-vector multiplication only
pub struct BMMNode<F, S, PCS> {
    /// The row-major flattened unpadded vector of weights
    weights: Vec<QSmallType>,
    /// The padded weight vector
    pub padded_weights: Vec<QSmallType>,
    /// The unpadded vector of biases
    bias: Vec<QLargeType>,
    /// The padded bias vector
    pub padded_bias: Vec<QLargeType>,
    /// Unpadded imensions (rows, columns)
    dims: (usize, usize),
    /// The logarithm of the padded dimensions (rows, columns)
    pub padded_dims_log: (usize, usize),
    /// Zero-point quantisation parameter of the input
    pub input_zero_point: QSmallType,

    phantom: PhantomData<(F, S, PCS)>,
}

/// Commitment to a BMM node, consisting of a commitment to the *dual* of the
/// weight MLE and one to the *dual* of the bias MLE
pub struct BMMNodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub weight_com: LabeledCommitment<PCS::Commitment>,
    pub bias_com: LabeledCommitment<PCS::Commitment>,
}

impl<F, S, PCS> Commitment for BMMNodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
}

/// Commitment states associated to a BMMNodeCommitment: one for the weight and
/// one for the bias
pub struct BMMNodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub weight_com_state: PCS::CommitmentState,
    pub bias_com_state: PCS::CommitmentState,
}

impl<F, S, PCS> CommitmentState for BMMNodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
}

/// Proof of execution of a BMM node, consisting of a sumcheck proof and four
/// PCS opening proofs
pub struct BMMNodeProof<
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
> {
    /// Sumcheck protocol proof for the polynomial
    /// g(x) = (input - zero_point)^(x) * W^(r, x),
    /// where v^ denotes the dual of the MLE of v and r is a challenge point
    pub sumcheck_proof: Proof<F>,

    /// Value of the *dual* of the input MLE at the challenge point s and proof
    /// of opening
    pub input_opening_proof: PCS::Proof,
    pub input_opening_value: F,

    /// Value of the *dual* of the weight MLE at the challenge point r || s and proof of
    /// opening
    pub weight_opening_proof: PCS::Proof,
    pub weight_opening_value: F,

    /// Proof of opening of the *duals* of the output and bias MLEs at the
    // challenge point
    pub output_bias_opening_proof: PCS::Proof,

    /// Value of the *dual* of the weight MLE at the challenge point and proof of
    /// opening
    pub output_opening_value: F,
    pub bias_opening_value: F,
}

impl<F, S, PCS> NodeOpsNative for BMMNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.dims.1]
    }

    fn evaluate(&self, input: &QTypeArray) -> QTypeArray {
        // Sanity checks
        // TODO systematise
        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("BMM node expects QSmallType as its QArray input type"),
        };

        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: BMM node expects a 1-dimensional input array"
        );
        assert_eq!(
            self.dims.0,
            input.len(),
            "Length mismatch: BMM node expects input with {} elements, got {} elements instead",
            self.dims.0,
            input.len()
        );

        let input: QArray<QLargeType> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = input - self.input_zero_point as QLargeType;

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

        let output = QArray::new(accumulators, vec![self.dims.1]);

        QTypeArray::L(output)
    }
}

impl<F, S, PCS> NodeOpsCommon<F, S, PCS> for BMMNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_dims_log.1]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_dims_log.0 + self.padded_dims_log.1
    }
}

impl<F, S, PCS> BMMNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub fn new(
        weights: Vec<QSmallType>,
        bias: Vec<QLargeType>,
        dims: (usize, usize),
        input_zero_point: QSmallType,
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

        let padded_dims_log: (usize, usize) = (
            log2(dims.0.next_power_of_two()) as usize,
            log2(dims.1.next_power_of_two()) as usize,
        );

        // Padding the weights
        let weight_array = QArray::new(weights.clone(), vec![dims.0, dims.1]);

        let padded_weights = weight_array
            .compact_resize(
                vec![dims.0.next_power_of_two(), dims.1.next_power_of_two()],
                0,
            )
            .move_values();

        // Padding the bias
        let mut padded_bias = bias.clone();
        padded_bias.resize(dims.1.next_power_of_two(), 0);

        Self {
            weights,
            padded_weights,
            bias,
            padded_bias,
            dims,
            padded_dims_log,
            input_zero_point,
            phantom: PhantomData,
        }
    }

    pub(crate) fn padded_dims_log(&self) -> (usize, usize) {
        self.padded_dims_log
    }

    pub(crate) fn input_zero_point(&self) -> QSmallType {
        self.input_zero_point
    }
}
// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
