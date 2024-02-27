use ark_std::marker::PhantomData;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

use crate::model::qarray::{QArray, QTypeArray};
use crate::model::Poly;
use crate::quantization::{requantise_fc, BMMQInfo, QInfo, QScaleType, QSmallType, RoundingScheme};
use crate::{Commitment, CommitmentState};

use super::{NodeOpsCommon, NodeOpsNative};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Apply requantisation after a BMM argument
pub struct RequantiseBMMNode<F, S, PCS> {
    // Number of units
    size: usize,

    // log2 of the number of units
    pub padded_size_log: usize,

    /// Quantisation info associated to the input BMM result
    pub q_info: BMMQInfo,

    phantom: PhantomData<(F, S, PCS)>,
}

pub struct RequantiseBMMNodeCommitment();

impl Commitment for RequantiseBMMNodeCommitment {}

pub struct RequantiseBMMNodeCommitmentState();

impl CommitmentState for RequantiseBMMNodeCommitmentState {}

pub struct RequantiseBMMNodeProof {
    // this will be the sumcheck proof
}

impl<F, S, PCS> NodeOpsNative for RequantiseBMMNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }

    fn evaluate(&self, input: &QTypeArray) -> QTypeArray {
        // Sanity checks
        // TODO systematise
        let input = match input {
            QTypeArray::L(i) => i,
            _ => panic!("RequantiseBMM node expects QLargeType as its QArray input type"),
        };

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

        let output: QArray<QSmallType> = requantise_fc(
            input.values(),
            &self.q_info,
            RoundingScheme::NearestTiesEven,
        )
        .into();

        QTypeArray::S(output)
    }
}

impl<F, S, PCS> NodeOpsCommon<F, S, PCS> for RequantiseBMMNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_size_log]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_size_log
    }
}

impl<F, S, PCS> RequantiseBMMNode<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub fn new(
        size: usize,
        s_i: QScaleType,
        z_i: QSmallType,
        s_w: QScaleType,
        z_w: QSmallType,
        s_o: QScaleType,
        z_o: QSmallType,
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
