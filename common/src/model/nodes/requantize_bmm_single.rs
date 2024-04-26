use ark_std::log2;

use crate::model::tensor::SmallNIO;
use crate::quantization::{quantize_multiplier, requantize_single_round};
use crate::{Commitment, CommitmentState, NIOTensor};

use super::{NodeOpsNative, NodeOpsPadded};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Apply requantization after a BMM argument
#[derive(Clone)]
pub struct RequantizeBMMSingleNode<ST: SmallNIO> {
    // Number of units
    size: usize,

    // log2 of the number of units
    pub padded_size_log: usize,

    // Represents a non-negative effective right shift
    full_shift: usize,

    //
    effective_multiplier: ST::LT,

    //
    output_zero_point: ST,
}

pub struct RequantizeBMMSingleNodeCommitment();

impl Commitment for RequantizeBMMSingleNodeCommitment {}

pub struct RequantizeBMMSingleNodeCommitmentState();

impl CommitmentState for RequantizeBMMSingleNodeCommitmentState {}

pub struct RequantizeBMMSingleNodeProof {
    // this will be the sumcheck proof
}

impl<ST> NodeOpsNative<ST> for RequantizeBMMSingleNode<ST>
where
    ST: SmallNIO,
{
    fn shape(&self) -> (Vec<usize>, Vec<usize>) {
        (vec![self.size], vec![self.size])
    }

    fn evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        let input = input.ref_large();

        // Sanity checks
        self.assert_valid_input(&input.shape());

        let output = requantize_single_round::<ST, ST::LT>(
            input.values(),
            self.effective_multiplier,
            self.full_shift,
            self.output_zero_point,
        )
        .into();

        NIOTensor::S(output)
    }

    fn type_name(&self) -> &'static str {
        "RequantizeBMMSingle"
    }

    fn com_num_vars(&self) -> usize {
        self.padded_size_log
    }
}

impl<ST> NodeOpsPadded<ST> for RequantizeBMMSingleNode<ST>
where
    ST: SmallNIO,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_size_log]
    }

    fn padded_shape(&self) -> (Vec<usize>, Vec<usize>) {
        let padded_size = 1 << self.padded_size_log;
        (vec![padded_size], vec![padded_size])
    }

    fn padded_evaluate(&self, input: &NIOTensor<ST>) -> NIOTensor<ST> {
        let input = input.ref_large();

        // Sanity checks
        self.assert_valid_input(input.shape());

        let output = requantize_single_round::<ST, ST::LT>(
            input.values(),
            self.effective_multiplier,
            self.full_shift,
            self.output_zero_point,
        )
        .into();

        NIOTensor::S(output)
    }
}

impl RequantizeBMMSingleNode<i8> {
    pub fn new(size: usize, s_i: f32, s_w: f32, s_o: f32, z_o: i8) -> Self {
        let padded_size_log = log2(size.next_power_of_two()) as usize;

        // cast scales to a type with higher precision
        let (s_i, s_w, s_o) = (s_i as f64, s_w as f64, s_o as f64);
        let double_multiplier = s_i * s_w / s_o;

        // compute full shift and effective multiplier
        let (effective_multiplier, effective_shift) = quantize_multiplier(double_multiplier);

        Self {
            size,
            padded_size_log,
            full_shift: effective_shift + (i32::BITS - 1) as usize,
            effective_multiplier,
            output_zero_point: z_o,
        }
    }
}
// TODO in constructor, add quantisation information checks? (e.g. z_weight = 0, etc.)
