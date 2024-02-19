use ark_std::rc::Rc;

use ark_poly::{MultilinearExtension, Polynomial};
use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};
use ark_std::log2;
use ark_std::rand::RngCore;
use ark_sumcheck::ml_sumcheck::protocol::ListOfProductsOfPolynomials;
use ark_sumcheck::ml_sumcheck::{MLSumcheck, Proof};

use crate::model::qarray::{QArray, QTypeArray};
use crate::model::{LabeledPoly, Poly};
use crate::quantization::{BMMQInfo, QInfo, QLargeType, QScaleType, QSmallType};
use crate::{Commitment, CommitmentState};

use super::{NodeCommitment, NodeCommitmentState, NodeOps, NodeOpsSNARK, NodeProof};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Start with 2D matrices, and Mat-by-vector multiplication only
pub(crate) struct BMMNode<F, S, PCS> {
    /// The row-major flattened unpadded vector of weights
    weights: Vec<QSmallType>,
    /// The padded weight vector
    padded_weights: Vec<QSmallType>,
    /// The unpadded vector of biases
    bias: Vec<QLargeType>,
    /// The padded bias vector
    padded_bias: Vec<QLargeType>,
    /// Unpadded imensions (rows, columns)
    dims: (usize, usize),
    /// The logarithm of the padded dimensions (rows, columns)
    padded_dims_log: (usize, usize),
    /// Zero-point quantisation parameter of the input
    input_zero_point: QSmallType,

    phantom: PhantomData<(F, S, PCS)>,
}

pub(crate) struct BMMNodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    weight_com: PCS::Commitment,
    bias_com: PCS::Commitment,
}

impl<F, S, PCS> Commitment for BMMNodeCommitment<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
}

pub(crate) struct BMMNodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    weight_com_state: PCS::CommitmentState,
    bias_com_state: PCS::CommitmentState,
}

impl<F, S, PCS> CommitmentState for BMMNodeCommitmentState<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
}

pub(crate) struct BMMNodeProof<
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
> {
    sumcheck_proof: Proof<F>,
    opening_proof: PCS::Proof,
    claimed_evaluations: Vec<F>,
}

impl<F, S, PCS> NodeOps for BMMNode<F, S, PCS>
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

impl<F, S, PCS> NodeOpsSNARK<F, S, PCS> for BMMNode<F, S, PCS>
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

    // This function naively computes entries which are known to be zero. It is
    // meant to exactly mirror the proof-system multiplication proved by the
    // sumcheck argument. Requantisation and shifting are also applied to these
    // trivial entries, as the proof system does.
    fn padded_evaluate(&self, input: &QTypeArray) -> QTypeArray {
        let input = match input {
            QTypeArray::S(i) => i,
            _ => panic!("BMM node expects QSmallType as its QArray input type"),
        };

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

        let input: QArray<QLargeType> = input.cast();

        // TODO this is a bigger question: can this overflow an i8? Supposedly the point of quantisation
        // is that input-by-weight products can be computed in i8. To be safe, let us use the large type here
        let shifted_input = input - self.input_zero_point as QLargeType;

        let mut accumulators = self.padded_bias.clone();

        // TODO this can be made more elegant (efficient?) using addition of QArrays after defining suitable operators

        // TODO since we have acumulators, this can be done more efficiently going row-wise to avoid re-caching the input
        for col in 0..padded_dims.1 {
            // TODO does the compiler realise it doesn't need to access accumulators[col] on every iteration of the inner loop? ow change
            for row in 0..padded_dims.0 {
                accumulators[col] += shifted_input[row]
                    * (self.padded_weights[row * padded_dims.1 + col] as QLargeType)
            }
        }

        let output = QArray::new(accumulators, vec![padded_dims.1]);

        QTypeArray::L(output)
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        // TODO should we separate the associated commitment type into one with state and one without?

        let num_vars_weights = self.padded_dims_log.0 + self.padded_dims_log.1;
        let padded_weights_f: Vec<F> = self.padded_weights.iter().map(|w| F::from(*w)).collect();

        let weight_poly = LabeledPolynomial::new(
            "weight_poly".to_string(),
            Poly::from_evaluations_vec(num_vars_weights, padded_weights_f),
            None,
            None,
        );

        let padded_bias_f: Vec<F> = self.padded_bias.iter().map(|b| F::from(*b)).collect();

        let bias_poly = LabeledPolynomial::new(
            "bias_poly".to_string(),
            Poly::from_evaluations_vec(self.padded_dims_log.1, padded_bias_f),
            None,
            None,
        );

        let coms = PCS::commit(ck, vec![&weight_poly, &bias_poly], rng).unwrap();

        (
            NodeCommitment::BMM(BMMNodeCommitment {
                weight_com: coms.0[0].commitment().clone(),
                bias_com: coms.0[1].commitment().clone(),
            }),
            NodeCommitmentState::BMM(BMMNodeCommitmentState {
                weight_com_state: coms.1[0].clone(),
                bias_com_state: coms.1[1].clone(),
            }),
        )
    }

    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        sponge: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input: LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: PCS::CommitmentState,
        output: LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        // we can squeeze directly, since the sponge has already absorbed all the
        // commitments in Model::prove_inference
        let r: Vec<F> = sponge.squeeze_field_elements(self.padded_dims_log.1);

        // TODO consider whether this can be done once and stored
        let weights_f = self.padded_weights.iter().map(|w| F::from(*w)).collect();
        // TODO this might need LE -> BE conversion
        let weights_mle = Poly::from_evaluations_vec(self.com_num_vars(), weights_f);

        // TODO we actually need fix_variables_last
        weights_mle.fix_variables(&r);

        // Constructing the sumcheck polynomial
        // big_poly(x) = input(x) * weights(x, r)
        let mut big_poly = ListOfProductsOfPolynomials::new(self.padded_dims_log.0);

        // TODO we are cloning the input here, can we do better?
        big_poly.add_product(
            vec![weights_mle, (*input).clone()]
                .into_iter()
                .map(Rc::new)
                .collect::<Vec<_>>(),
            F::one(),
        );

        let (sumcheck_proof, prover_state) =
            MLSumcheck::<F, S>::prove_as_subprotocol(&big_poly, sponge).unwrap();

        // Prover computes the claimed evaluations of Weights, Input, at the random point
        // Note this is a different random point than `r` above: `prover_state.randomness` is
        // the list of random values sampled vy V during the sumcheck itself
        let claimed_evaluations: Vec<F> = big_poly
            .flattened_ml_extensions
            .iter()
            .map(|x| x.evaluate(&prover_state.randomness))
            .collect();

        // TODO need to pass the labeled poly, and the commitment to, and the state for, the weights matrix. Currently only passing the data related to the input
        let opening_proof = PCS::open(
            &ck,
            &[input],
            &[(*input_com).clone()],
            &prover_state.randomness,
            sponge,
            &[input_com_state],
            None,
        )
        .unwrap();

        NodeProof::BMM(BMMNodeProof {
            sumcheck_proof,
            opening_proof,
            claimed_evaluations,
        })
    }
}

impl<F, S, PCS> BMMNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(
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
}
// TODO in constructor, add quantisation information checks? (s_bias = s_input * s_weight, z_bias = 0, z_weight = 0, etc.)
// TODO in constructor, check bias length matches appropriate matrix dimension
