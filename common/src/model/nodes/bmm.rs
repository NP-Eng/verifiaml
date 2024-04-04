use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::log2;

use ark_sumcheck::ml_sumcheck::Proof;

use crate::cp_snark::NodeCPSNARK;
use crate::model::qarray::{InnerType, QArray};
use crate::model::Poly;

use super::{NodeOpsNative, NodeOpsPadded};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Start with 2D matrices, and Mat-by-vector multiplication only
pub struct BMMNode<ST, LT> {
    /// The row-major flattened unpadded vector of weights
    weights: QArray<ST>,
    /// The padded weight vector
    pub padded_weights: QArray<ST>,
    /// The unpadded vector of biases
    bias: QArray<LT>,
    /// The padded bias vector
    pub padded_bias: QArray<LT>,
    /// Unpadded imensions (rows, columns)
    dims: (usize, usize),
    /// The logarithm of the padded dimensions (rows, columns)
    pub padded_dims_log: (usize, usize),
    /// Zero-point quantisation parameter of the input
    pub input_zero_point: ST,
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

impl<ST, LT> NodeOpsNative<ST, LT> for BMMNode<ST, LT>
where
    ST: InnerType,
    LT: InnerType + From<ST>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.dims.1]
    }

    fn evaluate(&self, input: &QArray<ST>) -> QArray<LT> {
        // Sanity checks
        // TODO systematise
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

        let input: QArray<LT> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = input - LT::from(self.input_zero_point);

        let mut accumulators = self.bias.values().clone();

        // TODO this can be made more elegant (efficient?) using addition of QArrays after defining suitable operators

        // TODO since we have acumulators, this can be done more efficiently going row-wise to avoid re-caching the input
        for col in 0..self.dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..self.dims.0 {
                accumulators[col] +=
                    shifted_input[row] * LT::from(self.weights[row * self.dims.1 + col])
            }
        }

        QArray::new(accumulators, vec![self.dims.1])
    }
}

impl<ST, LT> NodeOpsPadded<ST, LT> for BMMNode<ST, LT>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_dims_log.1]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_dims_log.0 + self.padded_dims_log.1
    }

    // This function naively computes entries which are known to be zero. It is
    // meant to exactly mirror the proof-system multiplication proved by the
    // sumcheck argument. Requantisation and shifting are also applied to these
    // trivial entries, as the proof system does.
    fn padded_evaluate(&self, input: &QArray<ST>) -> QArray<LT> {
        let padded_dims = (1 << self.padded_dims_log.0, 1 << self.padded_dims_log.1);

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: BMM node expects a 1-dimensional input array"
        );

        assert_eq!(
            padded_dims.0,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_dims.0,
            input.len()
        );

        let input: QArray<LT> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = input - LT::from(self.input_zero_point);

        let mut accumulators = self.padded_bias.values().clone();

        // TODO this can be made more elegant (efficient?) using addition of QArrays after defining suitable operators

        // TODO since we have acumulators, this can be done more efficiently going row-wise to avoid re-caching the input
        for col in 0..padded_dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..padded_dims.0 {
                accumulators[col] +=
                    shifted_input[row] * LT::from(self.padded_weights[row * padded_dims.1 + col])
            }
        }

        QArray::new(accumulators, vec![padded_dims.1])
    }
}

impl<ST, LT> BMMNode<ST, LT>
where
    ST: InnerType,
    LT: InnerType,
{
    pub fn new(weights: QArray<ST>, bias: QArray<LT>, input_zero_point: ST) -> Self {
        let dims = (weights.shape()[0], weights.shape()[1]);

        assert_eq!(
            bias.len(),
            dims.1,
            "Bias vector length does not match the number of columns"
        );

        let padded_dims_log: (usize, usize) = (
            log2(dims.0.next_power_of_two()) as usize,
            log2(dims.1.next_power_of_two()) as usize,
        );

        // Padding the weights and bias
        let padded_weights = weights.clone().compact_resize(
            vec![dims.0.next_power_of_two(), dims.1.next_power_of_two()],
            ST::ZERO,
        );

        let padded_bias = bias
            .clone()
            .compact_resize(vec![dims.1.next_power_of_two()], LT::ZERO);

        Self {
            weights,
            padded_weights,
            bias,
            padded_bias,
            dims,
            padded_dims_log,
            input_zero_point,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn padded_dims_log(&self) -> (usize, usize) {
        self.padded_dims_log
    }

    #[allow(dead_code)]
    pub(crate) fn input_zero_point(&self) -> ST {
        self.input_zero_point
    }
}
// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension

pub struct BMMParamCommitment<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    padded_weight_com: LabeledCommitment<PCS::Commitment>,
    padded_bias_com: LabeledCommitment<PCS::Commitment>,
}

pub struct BMMParamHint<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    padded_weight_hint: PCS::CommitmentState,
    padded_bias_hint: PCS::CommitmentState,
}

pub struct BMMParamValue<F: PrimeField> {
    padded_weights: Vec<F>,
    padded_bias: Vec<F>,
}

pub struct BMMEvaluationKey<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    key: PCS::CommitterKey,
    padded_dims_log: (usize, usize),
}

pub struct BMMVerificationKey<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    key: PCS::VerifierKey,
    padded_dims_log: (usize, usize),
}

pub struct BMMInstance<F: PrimeField> {
    input_zero_point: F,
}

pub struct BMMProof<
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

impl<F, S, PCS, ST, LT> NodeCPSNARK<F, S, PCS> for BMMNode<ST, LT>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    type CommitmentKey = PCS::CommitterKey;
    type EvaluationKey = BMMEvaluationKey<F, S, PCS>;
    type VerificationKey = BMMVerificationKey<F, S, PCS>;

    type ParamCommitment = BMMParamCommitment<F, S, PCS>;
    type ParamHint = BMMParamHint<F, S, PCS>;
    type ParamValue = BMMParamValue<F>;

    type Instance = BMMInstance<F>;

    type Proof = BMMProof<F, S, PCS>;

    fn key_gen(
        ck: &Self::CommitmentKey,
        node: &Self,
    ) -> (Self::EvaluationKey, Self::VerificationKey) {
        unimplemented!()
    }

    fn prove(
        ek: &Self::EvaluationKey,
        instance: &Self::Instance,
        param_commitment: &Self::ParamCommitment,
        input_commitment: &PCS::Commitment,
        output_commitment: &PCS::Commitment,
        param_value: &Self::ParamValue,
        input_value: &Vec<F>,
        output_value: &Vec<F>,
        param_hint: &Self::ParamHint,
        input_hint: &PCS::CommitmentState,
        output_hint: &PCS::CommitmentState,
        // no non-committed witness omega
    ) -> Self::Proof {
        unimplemented!()
    }

    fn verify_proof(
        vk: &Self::VerificationKey,
        instance: &Self::Instance,
        param_commitment: &Self::ParamCommitment,
        input_commitment: &PCS::Commitment,
        output_commitment: &PCS::Commitment,
        pi: &Self::Proof,
    ) -> bool {
        unimplemented!()
    }
}
