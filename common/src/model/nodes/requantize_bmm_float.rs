use std::any::Any;

use ark_std::log2;

use crate::model::tensor::{SmallNIO, Tensor};
use crate::quantization::{requantize_fc, BMMQInfo, QInfo, QScaleType, RoundingScheme};
use crate::{Commitment, CommitmentState, NIOTensor};

use super::{NodeOpsNative, NodeOpsPadded};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Apply requantization after a BMM argument
pub struct RequantizeBMMFloatNode<ST> {
    // Number of units
    size: usize,

    // log2 of the number of units
    pub padded_size_log: usize,

    /// Quantisation info associated to the input BMM result
    pub q_info: BMMQInfo<ST>,
}

pub struct RequantizeBMMNodeCommitment();

impl Commitment for RequantizeBMMNodeCommitment {}

pub struct RequantizeBMMNodeCommitmentState();

impl CommitmentState for RequantizeBMMNodeCommitmentState {}

pub struct RequantizeBMMNodeProof {
    // this will be the sumcheck proof
}

impl<ST> NodeOpsNative<ST> for RequantizeBMMFloatNode<ST>
where
    ST: SmallNIO,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }

    fn evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        // Sanity checks
        // TODO systematise
        let input = input.ref_large();

        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantizeBMM node expects a 1-dimensional input array"
        );
        assert_eq!(
            self.size,
            input.len(),
            "Length mismatch: RequantizeBMM node expects input with {} elements, got {} elements instead",
            self.size,
            input.len()
        );

        let output: Tensor<ST> = requantize_fc::<ST, ST::LT>(
            input.values(),
            &self.q_info,
            RoundingScheme::NearestTiesEven,
        )
        .into();

        NIOTensor::S(output)
    }

    fn type_name(&self) -> &'static str {
        "RequantizeBMMFloat"
    }

    fn com_num_vars(&self) -> usize {
        self.padded_size_log
    }
}

impl<ST> NodeOpsPadded<ST> for RequantizeBMMFloatNode<ST>
where
    ST: SmallNIO,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_size_log]
    }

    fn padded_evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        let padded_size = 1 << self.padded_size_log;
        let input = input.ref_large();

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantizeBMM node expects a 1-dimensional input array"
        );

        assert_eq!(
            padded_size,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_size,
            input.len()
        );

        let output: Tensor<ST> = requantize_fc::<ST, ST::LT>(
            input.values(),
            &self.q_info,
            RoundingScheme::NearestTiesEven,
        )
        .into();

        NIOTensor::S(output)
    }
}

impl<ST> RequantizeBMMFloatNode<ST> {
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
