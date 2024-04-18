use ark_std::log2;

use crate::model::tensor::{Numeric, Tensor};
use crate::quantization::{quantize_multiplier, requantize_ref};
use crate::{Commitment, CommitmentState};

use super::{NodeOpsNative, NodeOpsPadded};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Apply requantization after a BMM argument
pub struct RequantizeBMMRefNode<ST, LT> {
    // Number of units
    size: usize,

    // log2 of the number of units
    pub padded_size_log: usize,

    // Represents a non-negative right shift reduced by the implicit
    // fixed-point-arithmetic shift in DoublingHighRoundMultiply
    effective_shift: usize,

    //
    effective_multiplier: LT,

    //
    output_zero_point: ST,
}

pub struct RequantizeBMMRefNodeCommitment();

impl Commitment for RequantizeBMMRefNodeCommitment {}

pub struct RequantizeBMMRefNodeCommitmentState();

impl CommitmentState for RequantizeBMMRefNodeCommitmentState {}

pub struct RequantizeBMMRefNodeProof {
    // this will be the sumcheck proof
}

impl<ST, LT> NodeOpsNative<LT, ST> for RequantizeBMMRefNode<ST, LT>
where
    ST: Numeric + TryFrom<LT>,
    LT: Numeric + From<ST>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }

    fn evaluate(&self, input: &Tensor<LT>) -> Tensor<ST> {
        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantizeBMMRef node expects a 1-dimensional input array"
        );
        assert_eq!(
            self.size,
            input.len(),
            "Length mismatch: RequantizeBMMRef node expects input with {} elements, got {} elements instead",
            self.size,
            input.len()
        );

        let output: Tensor<ST> = requantize_ref::<ST, LT>(
            input.values(),
            self.effective_multiplier,
            self.effective_shift,
            self.output_zero_point,
        )
        .into();

        output
    }
}

impl<ST, LT> NodeOpsPadded<LT, ST> for RequantizeBMMRefNode<ST, LT>
where
    ST: Numeric + TryFrom<LT>,
    LT: Numeric + From<ST>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_size_log]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_size_log
    }

    fn padded_evaluate(&self, input: &Tensor<LT>) -> Tensor<ST> {
        let padded_size = 1 << self.padded_size_log;

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantizeBMMRef node expects a 1-dimensional input array"
        );

        assert_eq!(
            padded_size,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_size,
            input.len()
        );

        let output: Tensor<ST> = requantize_ref::<ST, LT>(
            input.values(),
            self.effective_multiplier,
            self.effective_shift,
            self.output_zero_point,
        )
        .into();
        output
    }
}

impl RequantizeBMMRefNode<i8, i32> {
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
            effective_shift,
            effective_multiplier,
            output_zero_point: z_o,
        }
    }
}
// TODO in constructor, add quantisation information checks? (e.g. z_weight = 0, etc.)
