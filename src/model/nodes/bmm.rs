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
    weight_com: LabeledCommitment<PCS::Commitment>,
    bias_com: LabeledCommitment<PCS::Commitment>,
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
    input_opening_proof: PCS::Proof,
    input_opening_value: F,
    weight_opening_proof: PCS::Proof,
    weight_opening_value: F,
    bias_opening_proof: PCS::Proof,
    bias_opening_value: F,
    output_opening_proof: PCS::Proof,
    output_opening_value: F,
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
        let padded_weights_f: Vec<F> = self.padded_weights.iter().map(|w| F::from(*w)).collect();

        // TODO part of this code is duplicated in prove, another hint that this should probs
        // be stored
        let weight_poly = LabeledPolynomial::new(
            "weight_poly".to_string(),
            Poly::from_evaluations_vec(self.com_num_vars(), padded_weights_f),
            Some(1),
            None,
        );

        let padded_bias_f: Vec<F> = self.padded_bias.iter().map(|b| F::from(*b)).collect();

        let bias_poly = LabeledPolynomial::new(
            "bias_poly".to_string(),
            Poly::from_evaluations_vec(self.padded_dims_log.1, padded_bias_f),
            Some(1),
            None,
        );

        let coms = PCS::commit(ck, vec![&weight_poly, &bias_poly], rng).unwrap();

        (
            NodeCommitment::BMM(BMMNodeCommitment {
                weight_com: coms.0[0].clone(),
                bias_com: coms.0[1].clone(),
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
        node_com_state: &NodeCommitmentState<F, S, PCS>,
        input: &LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: &PCS::CommitmentState,
        output: &LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        let (weight_com, bias_com) = match node_com {
            NodeCommitment::BMM(BMMNodeCommitment {
                weight_com,
                bias_com,
            }) => (weight_com, bias_com),
            _ => panic!("BMMNode::prove expected node commitment of type BMMNodeCommitment"),
        };

        let (weight_com_state, bias_com_state) = match node_com_state {
            NodeCommitmentState::BMM(BMMNodeCommitmentState {
                weight_com_state,
                bias_com_state,
            }) => (weight_com_state, bias_com_state),
            _ => panic!(
                "BMMNode::prove expected node commitment state of type BMMNodeCommitmentState"
            ),
        };

        // we can squeeze directly, since the sponge has already absorbed all the
        // commitments in Model::prove_inference
        let r: Vec<F> = sponge.squeeze_field_elements(self.padded_dims_log.1);

        let input_mle = input.polynomial().clone();

        // TODO consider whether this can be done once and stored
        let weights_f = self.padded_weights.iter().map(|w| F::from(*w)).collect();
        // TODO this might need LE -> BE conversion
        let weight_mle = Poly::from_evaluations_vec(self.com_num_vars(), weights_f);

        // TODO consider whether this can be done once and stored
        let bias_f = self.padded_bias.iter().map(|w| F::from(*w)).collect();
        // TODO this might need LE -> BE conversion
        let bias_mle = Poly::from_evaluations_vec(self.padded_dims_log.1, bias_f);

        // TODO is output_opening_value directly available from the output of sumcheck?
        // It doesn't need to be used until the end of the method
        let bias_opening_value = bias_mle.evaluate(&r);
        let output_opening_value = output.evaluate(&r);

        // TODO we actually need fix_variables_last
        let bound_weight_mle = weight_mle.fix_variables(&r);

        // Constructing the sumcheck polynomial
        // big_poly(x) := input(x) * weights(x, r)
        let mut big_poly = ListOfProductsOfPolynomials::new(self.padded_dims_log.0);

        // TODO we are cloning the input here, can we do better?
        big_poly.add_product(
            vec![input_mle, bound_weight_mle]
                .into_iter()
                .map(Rc::new)
                .collect::<Vec<_>>(),
            F::one(),
        );

        let (sumcheck_proof, prover_state) =
            MLSumcheck::<F, S>::prove_as_subprotocol(&big_poly, sponge).unwrap();

        // The prover computes the claimed evaluations of weight_mle and
        // input_mle at the random challenge point
        // s:= `prover_state.randomness`, the list of random values sampled by
        // the verifier duriing sumcheck. Note that this is different from `r`
        // above.
        //
        // We need to open input_mle(s) * weight_mle(s, r) as well as
        // output_mle(r)
        let claimed_evaluations: Vec<F> = big_poly
            .flattened_ml_extensions
            .iter()
            .map(|x| x.evaluate(&prover_state.randomness))
            .collect();

        let input_opening_proof = PCS::open(
            &ck,
            [input],
            [input_com],
            &prover_state.randomness,
            sponge,
            [input_com_state],
            None,
        )
        .unwrap();

        // TODO remove
        println!("*** After opening input ***");

        let weight_opening_proof = PCS::open(
            &ck,
            [&LabeledPolynomial::new(
                "weight_mle".to_string(),
                weight_mle,
                Some(1),
                None,
            )],
            [weight_com],
            &prover_state
                .randomness
                .into_iter()
                .chain(r.clone().into_iter())
                .collect(),
            sponge,
            [weight_com_state],
            None,
        )
        .unwrap();

        let bias_opening_proof = PCS::open(
            &ck,
            [&LabeledPolynomial::new(
                "bias_mle".to_string(),
                bias_mle,
                Some(1),
                None,
            )],
            [bias_com],
            &r,
            sponge,
            [bias_com_state],
            None,
        )
        .unwrap();

        let output_opening_proof = PCS::open(
            &ck,
            [output],
            [output_com],
            &r,
            sponge,
            [output_com_state],
            None,
        )
        .unwrap();

        NodeProof::BMM(BMMNodeProof {
            sumcheck_proof,
            input_opening_proof,
            input_opening_value: claimed_evaluations[0],
            weight_opening_proof,
            weight_opening_value: claimed_evaluations[1],
            bias_opening_proof,
            bias_opening_value,
            output_opening_proof,
            output_opening_value,
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
