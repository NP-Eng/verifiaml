use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, LabeledPolynomial, PolynomialCommitment};
use ark_std::fmt::Debug;
use ark_std::log2;
use ark_std::rand::RngCore;

use crate::model::qarray::{InnerType, QArray, QTypeArray};
use crate::model::{LabeledPoly, Poly};
use crate::quantization::{requantise_fc, BMMQInfo, QInfo, QScaleType, RoundingScheme};
use crate::{Commitment, CommitmentState};

use super::{NodeCommitment, NodeCommitmentState, NodeOps, NodeOpsSNARK, NodeProof};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Apply requantisation after a BMM argument
pub(crate) struct RequantiseBMMNode<F, S, PCS, ST>
where
    ST: InnerType,
{
    // Number of units
    size: usize,

    // log2 of the number of units
    padded_size_log: usize,

    /// Quantisation info associated to the input BMM result
    q_info: BMMQInfo<ST>,

    phantom: PhantomData<(F, S, PCS, ST)>,
}

pub(crate) struct RequantiseBMMNodeCommitment();

impl Commitment for RequantiseBMMNodeCommitment {}

pub(crate) struct RequantiseBMMNodeCommitmentState();

impl CommitmentState for RequantiseBMMNodeCommitmentState {}

pub(crate) struct RequantiseBMMNodeProof {
    // this will be the sumcheck proof
}

impl<F, S, PCS, ST, LT> NodeOps<ST, LT> for RequantiseBMMNode<F, S, PCS, ST>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }

    fn evaluate(&self, input: &QTypeArray<ST, LT>) -> QTypeArray<ST, LT> {
        // Sanity checks
        // TODO systematise
        let input = input.ref_large();

        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantiseBMM node expects a 1-dimensional input array"
        );
        assert_eq!(
            self.size,
            input.len(),
            "Length mismatch: RequantiseBMM node expects input with {} elements, got {} elements instead",
            self.size,
            input.len()
        );

        let output: QArray<ST> = requantise_fc(
            &input.values(),
            &self.q_info,
            RoundingScheme::NearestTiesEven,
        )
        .into();

        QTypeArray::S(output)
    }
}

impl<F, S, PCS, ST, LT> NodeOpsSNARK<F, S, PCS, ST, LT> for RequantiseBMMNode<F, S, PCS, ST>
where
    F: PrimeField + Absorb + From<ST> + From<LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType + TryFrom<LT>,
    <ST as TryFrom<LT>>::Error: Debug,
    LT: InnerType + From<ST>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_size_log]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_size_log
    }

    fn padded_evaluate(&self, input: &QTypeArray<ST, LT>) -> QTypeArray<ST, LT> {
        let input = input.ref_large();

        let padded_size = 1 << self.padded_size_log;

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantiseBMM node expects a 1-dimensional input array"
        );

        assert_eq!(
            padded_size,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_size,
            input.len()
        );

        let output: QArray<ST> = requantise_fc(
            input.values(),
            &self.q_info,
            RoundingScheme::NearestTiesEven,
        )
        .into();

        QTypeArray::S(output)
    }

    fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>) {
        (
            NodeCommitment::RequantiseBMM(RequantiseBMMNodeCommitment()),
            NodeCommitmentState::RequantiseBMM(RequantiseBMMNodeCommitmentState()),
        )
    }

    fn prove(
        &self,
        ck: &PCS::CommitterKey,
        s: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        node_com_state: &NodeCommitmentState<F, S, PCS>,
        input: &LabeledPoly<F>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        input_com_state: &PCS::CommitmentState,
        output: &LabeledPoly<F>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        output_com_state: &PCS::CommitmentState,
    ) -> NodeProof<F, S, PCS> {
        NodeProof::RequantiseBMM(RequantiseBMMNodeProof {})
    }
}

impl<F, S, PCS, ST> RequantiseBMMNode<F, S, PCS, ST>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType,
{
    pub(crate) fn new(
        size: usize,
        s_i: QScaleType,
        z_i: ST,
        s_w: QScaleType,
        z_w: ST,
        s_o: QScaleType,
        z_o: ST,
    ) -> Self {
        let padded_size_log = log2(size.next_power_of_two()) as usize;

        // TODO not all of these are needed
        let q_info = BMMQInfo {
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
            size,
            padded_size_log,
            q_info,
            phantom: PhantomData,
        }
    }
}
// TODO in constructor, add quantisation information checks? (e.g. z_weight = 0, etc.)
