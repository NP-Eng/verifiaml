use ark_std::{fmt::Debug, log2};

use crate::model::qarray::{InnerType, QArray, QTypeArray};
use crate::quantization::{requantise_fc, BMMQInfo, QInfo, QScaleType, RoundingScheme};
use crate::{Commitment, CommitmentState};

use super::{NodeOpsCommon, NodeOpsNative};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Apply requantisation after a BMM argument
pub struct RequantiseBMMNode<ST> {
    // Number of units
    size: usize,

    // log2 of the number of units
    pub padded_size_log: usize,

    /// Quantisation info associated to the input BMM result
    pub q_info: BMMQInfo<ST>,
}

pub struct RequantiseBMMNodeCommitment();

impl Commitment for RequantiseBMMNodeCommitment {}

pub struct RequantiseBMMNodeCommitmentState();

impl CommitmentState for RequantiseBMMNodeCommitmentState {}

pub struct RequantiseBMMNodeProof {
    // this will be the sumcheck proof
}

impl<ST, LT> NodeOpsNative<ST, LT> for RequantiseBMMNode<ST>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
    <ST as TryFrom<LT>>::Error: Debug,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }

    fn evaluate(&self, input: &QTypeArray<ST, LT>) -> QTypeArray<ST, LT> {
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

        let output: QArray<ST> = requantise_fc(
            input.values(),
            &self.q_info,
            RoundingScheme::NearestTiesEven,
        )
        .into();

        QTypeArray::S(output)
    }
}

impl<ST> NodeOpsCommon for RequantiseBMMNode<ST> {
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_size_log]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_size_log
    }
}

impl<ST> RequantiseBMMNode<ST> {
    pub fn new(
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
        }
    }
}
// TODO in constructor, add quantisation information checks? (e.g. z_weight = 0, etc.)
