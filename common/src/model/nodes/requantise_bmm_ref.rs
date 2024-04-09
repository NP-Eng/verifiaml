use ark_ff::Zero;
use ark_std::log2;
use std::marker::PhantomData;

use crate::model::qarray::{InnerType, QArray};
use crate::quantization::requantise_ref;
use crate::{Commitment, CommitmentState};

use super::{NodeOpsNative, NodeOpsPadded};

// TODO convention: input, bias and output are rows, the op is vec-by-mat (in that order)

/// Apply requantisation after a BMM argument
pub struct RequantiseBMMRefNode<ST, LT, XT> {
    // Number of units
    size: usize,

    // log2 of the number of units
    pub padded_size_log: usize,

    // Represents a non-negative right shift
    effective_shift: usize,

    //
    effective_multiplier: LT,

    //
    output_zero_point: ST,

    //
    extended_type: PhantomData<XT>,
}

pub struct RequantiseBMMRefNodeCommitment();

impl Commitment for RequantiseBMMRefNodeCommitment {}

pub struct RequantiseBMMRefNodeCommitmentState();

impl CommitmentState for RequantiseBMMRefNodeCommitmentState {}

pub struct RequantiseBMMRefNodeProof {
    // this will be the sumcheck proof
}

impl<ST, LT, XT> NodeOpsNative<LT, ST, XT> for RequantiseBMMRefNode<ST, LT, XT>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST> + From<u32>,
    XT: InnerType + From<LT>,
{
    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }

    fn evaluate(&self, input: &QArray<LT>) -> QArray<ST> {
        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantiseBMMRef node expects a 1-dimensional input array"
        );
        assert_eq!(
            self.size,
            input.len(),
            "Length mismatch: RequantiseBMMRef node expects input with {} elements, got {} elements instead",
            self.size,
            input.len()
        );

        let output: QArray<ST> = requantise_ref::<ST, LT, XT>(
            input.values(),
            self.effective_multiplier,
            self.effective_shift,
            self.output_zero_point,
        )
        .into();

        output
    }
}

impl<ST, LT, XT> NodeOpsPadded<LT, ST, XT> for RequantiseBMMRefNode<ST, LT, XT>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST> + From<u32>,
    XT: InnerType + From<LT>,
{
    fn padded_shape_log(&self) -> Vec<usize> {
        vec![self.padded_size_log]
    }

    fn com_num_vars(&self) -> usize {
        self.padded_size_log
    }

    fn padded_evaluate(&self, input: &QArray<LT>) -> QArray<ST> {
        let padded_size = 1 << self.padded_size_log;

        // Sanity checks
        // TODO systematise
        assert_eq!(
            input.num_dims(),
            1,
            "Incorrect shape: RequantiseBMMRef node expects a 1-dimensional input array"
        );

        assert_eq!(
            padded_size,
            input.len(),
            "Length mismatch: Padded fully connected node expected input with {} elements, got {} elements instead",
            padded_size,
            input.len()
        );

        let output: QArray<ST> = requantise_ref::<ST, LT, XT>(
            input.values(),
            self.effective_multiplier,
            self.effective_shift,
            self.output_zero_point,
        )
        .into();
        output
    }
}

impl RequantiseBMMRefNode<i8, i32, i64> {
    pub fn new(size: usize, s_i: f32, s_w: f32, s_o: f32, z_o: i8) -> Self {
        let padded_size_log = log2(size.next_power_of_two()) as usize;

        // cast scales to a type with higher precision
        let (s_i, s_w, s_o) = (s_i as f64, s_w as f64, s_o as f64);
        let double_multiplier = (s_i * s_w / s_o) as f64;

        // compute effective shift and effective multiplier
        let (effective_multiplier, effective_shift) = quantize_multiplier(double_multiplier);

        Self {
            size,
            padded_size_log,
            effective_shift,
            effective_multiplier,
            output_zero_point: z_o,
            extended_type: PhantomData::<i64>,
        }
    }
}
// TODO in constructor, add quantisation information checks? (e.g. z_weight = 0, etc.)

//
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/quantization_util.cc#L53-L104
fn quantize_multiplier(double_multiplier: f64) -> (i32, usize) {
    if double_multiplier.is_zero() {
        return (0, 0);
    }

    let (q, expon) = frexp(double_multiplier);

    assert!(expon < 0, "expon should be negative. Got: {}", expon);

    // Negate expon to obtain the number of right-shift bits
    let mut shift: usize = -expon as usize;

    // TF Lite uses C++'s round function under the hood as can be seen here:
    // https://github.com/tensorflow/tensorflow/blob/46f028f94dcd974705cd14e8abf05b9bd8f20bf0/tensorflow/lite/kernels/internal/cppmath.h#L35
    // The function rounds to the nearest integer, breaking ties away from zero.
    // The same strategy is implemented in Rust's round method:
    // https://doc.rust-lang.org/std/primitive.f64.html#method.round
    // See also: https://en.cppreference.com/w/c/numeric/fenv/FE_round
    let mut q_fixed = (q * ((1 << (i32::BITS - 1)) as f64)).round() as i64; // q * (1 << 31)

    // TFLITE_CHECK(q_fixed <= (1LL << 31));
    if q_fixed > 1 << i32::BITS - 1 {
        panic!(
            "q_fixed must not exceed {}. Got: {}",
            i32::BITS - 1,
            q_fixed
        );
    }

    if q_fixed == (1 << (i32::BITS - 1)) {
        // 1 << 31
        q_fixed /= 2;
        shift += 1;
    }

    // TFLITE_CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
    if q_fixed > i32::MAX as i64 {
        panic!("q_fixed must not exceed {}. Got: {}", i32::MAX, q_fixed);
    }

    // If exponent is too small.
    if (-expon as u32) < i32::BITS - 1 {
        // expon < -31
        shift = 0;
        q_fixed = 0;
    }

    (q_fixed as i32, shift)
}

// This function returns the normalized fraction and exponent of a double-precision number x.
// If the argument x is not zero, the normalized fraction is x times a power of two, and its
// absolute value is always in the range 0.5 (inclusive) to 1 (exclusive). If x is zero, then
// the normalized fraction and exponent should be zero. However, for our purposes, x should
// always be positive.
fn frexp(x: f64) -> (f64, isize) {
    const F64_EXPONENT_SHIFT: u64 = 52;
    const F64_EXPONENT_BIAS: u64 = 1023;
    const F64_EXPONENT_MASK: u64 = 0x7ff0000000000000;
    const F64_FRACTION_MASK: u64 = 0x000fffffffffffff;

    assert!(x > 0.0);

    let x_bits: u64 = x.to_bits();

    // truncate low order bits to compute the biased exponent
    let mut expon = ((x_bits & F64_EXPONENT_MASK) / (1 << F64_EXPONENT_SHIFT)) as i32;

    // assert 0 < expon < 1023<<1
    assert!(expon > 0 && expon < ((F64_EXPONENT_BIAS as i32) << 1));

    // unbias exponent
    expon -= (F64_EXPONENT_BIAS as i32) + 1;

    let mantissa = x_bits & F64_FRACTION_MASK;

    let q = ((mantissa + (1 << F64_EXPONENT_SHIFT)) as f64)
        / ((1_u64 << (F64_EXPONENT_SHIFT + 1)) as f64);

    (q, expon as isize)
}
