#[cfg(test)]
pub mod tests;

use ark_std::Zero;

use crate::model::tensor::Integral;

const F64_EXPONENT_SHIFT: u64 = 52;
const F64_EXPONENT_BIAS: u64 = 1023;
const F64_EXPONENT_MASK: u64 = 0x7ff0000000000000;
const F64_FRACTION_MASK: u64 = 0x000fffffffffffff;

// TODO if we decide to make the model generic on the quantisation process
// types (which is probably correct now that the qtypes are generics), these
// will go away
// Type for quantisation scales
pub(crate) type QScaleType = f32;
// Larger precision type to compute the requantization scale in some schemes
pub(crate) type QScaleComputationType = f64;

pub struct QInfo<ST> {
    pub scale: QScaleType,
    pub zero_point: ST,
}

// TODO: this will probably change to inference-ready requantization info
// Even what is being done now could be optimised by precomputing outside the
// evaluate function
pub struct BMMQInfo<ST> {
    pub input_info: QInfo<ST>,
    pub weight_info: QInfo<ST>,
    // Bias requantization information is not used (and is indeed directly
    // computable from the two above)
    pub output_info: QInfo<ST>,
}

// Strategies to requantize the output of a BMM node
#[derive(Debug, Clone, Copy)]
pub enum BMMRequantizationStrategy {
    Floating,  // Core: multiply the input by the floating-point scale
    Reference, // Core: fixed-point-multiply the input by the quantised
    //       scale, then right-shift further (and round)
    SingleRound, // Core: integer-multiply the input by the quantised scale,
                 //       then apply a single right shift (and round)
}

pub enum RoundingScheme {
    NearestTiesAwayFromZero,
    NearestTiesEven,
}

pub fn requantize_fc<ST: Integral, LT: Integral>(
    output: &[LT],
    q_info: &BMMQInfo<ST>,
    scheme: RoundingScheme,
) -> Vec<ST>
where
    ST: Integral + TryFrom<LT>,
    LT: Integral + From<ST>,
{
    match scheme {
        RoundingScheme::NearestTiesAwayFromZero => requantize_fc_ntafz::<ST, LT>(output, q_info),
        RoundingScheme::NearestTiesEven => requantize_fc_nte::<ST, LT>(output, q_info),
    }
}

fn requantize_fc_ntafz<ST, LT>(output: &[LT], q_info: &BMMQInfo<ST>) -> Vec<ST>
where
    ST: Integral + TryFrom<LT>,
    LT: Integral + From<ST>,
{
    // 1. Computing scale
    // TODO In actual schemes, this will be decomposed as (int, shift)
    let (s_i, s_w, s_o) = (
        q_info.input_info.scale,
        q_info.weight_info.scale,
        q_info.output_info.scale,
    );
    let (s_i, s_w, s_o) = (
        s_i as QScaleComputationType,
        s_w as QScaleComputationType,
        s_o as QScaleComputationType,
    );
    let s = (s_i * s_w / s_o) as QScaleType;

    // 2. Requantize
    // TODO add rayon for parallelisation?
    output
        .iter()
        .map(|x| {
            let x = LT::to_qscaletype(x) * s;
            let mut x = LT::from_qscaletype(x.round());
            x += LT::from(q_info.output_info.zero_point);
            ST::try_from(partial_ord_clamp(x, LT::from(ST::MIN), LT::from(ST::MAX)))
                .map_err(|_| "Unable to convert Large Type to Small Type")
                .unwrap()
        })
        .collect()
}

// The (unstable) method clamp comes from the trait Ord, which we cannot
// restrict Integral to as we need f32 to implement the latter. Note that this
// method is not meaningfully defined for classes that genuinely do not
// implement Ord (total order relation) but only PartialOrd (partial order
// relation).
fn partial_ord_clamp<T: PartialOrd>(x: T, min: T, max: T) -> T {
    if x <= min {
        min
    } else if x >= max {
        max
    } else {
        x
    }
}

fn requantize_fc_nte<ST: Integral, LT: Integral>(output: &[LT], q_info: &BMMQInfo<ST>) -> Vec<ST>
where
    ST: Integral + TryFrom<LT>,
    LT: Integral + From<ST>,
{
    // 1. Computing scale
    // TODO In actual schemes, this will be decomposed as (int, shift)
    let (s_i, s_w, s_o) = (
        q_info.input_info.scale,
        q_info.weight_info.scale,
        q_info.output_info.scale,
    );
    let (s_i, s_w, s_o) = (
        s_i as QScaleComputationType,
        s_w as QScaleComputationType,
        s_o as QScaleComputationType,
    );
    let s = (s_i * s_w / s_o) as QScaleType;

    // 2. Requantize
    // TODO add rayon for parallelisation?
    output
        .iter()
        .map(|x| {
            let x = LT::to_qscaletype(x) * s;
            let mut x = LT::from_qscaletype(x.round_ties_even()); // TODO which type to pick here? Should we check for overflows?
            x += LT::from(q_info.output_info.zero_point);
            ST::try_from(partial_ord_clamp(x, LT::from(ST::MIN), LT::from(ST::MAX)))
                .map_err(|_| "Unable to convert Large Type to Small Type")
                .unwrap()
        })
        .collect()
}

// Implementation of TF Lite's reference requantization.
pub fn requantize_ref<ST, LT>(
    // TODO Think whether we can afford to pass ownership here and change the iter() below by into_iter()
    output: &[LT],
    effective_multiplier: LT,
    effective_shift: usize,
    output_zero_point: ST,
) -> Vec<ST>
where
    ST: Integral + TryFrom<LT>,
    LT: Integral + From<ST>,
{
    // Computing auxiliary constants used for every input
    let effective_multiplier = LT::Double::from(effective_multiplier);
    let output_zero_point = LT::from(output_zero_point);

    // TODO: Add associated constant MAX_PLUS_ONE to Integral.
    let pow2_bits_minus_one = LT::ONE_DOUBLE << (LT::BITS - 1);

    // NOTE: Notice that they are independent of the input. Perhaps it is meaningful to turn:
    // xt_pow2_bits_minus_one, non_neg_nudge, and neg_nudge
    // into associated constants of type LT in order to avoid their recomputation per call?

    // TODO: After splitting InnerType, rewrite pow2 to use << instead of *.

    // Mask consists of full_shift ones
    let mask = (LT::ONE << effective_shift) - LT::ONE; // TODO: may overflow for some exponents
    let mask_div2 = mask >> 1;

    // Constants used during nudging
    let non_neg_nudge = LT::ONE_DOUBLE << (LT::BITS - 2);
    let neg_nudge = LT::ONE_DOUBLE - non_neg_nudge;

    // Requantize
    // TODO add rayon for parallelization?
    output
        .iter()
        .map(|x| {
            let (is_negative, nudge) = if *x >= LT::ZERO {
                (LT::ZERO, non_neg_nudge)
            } else {
                (LT::ONE, neg_nudge)
            };

            let product = LT::Double::from(*x) * effective_multiplier;

            let product_high: LT = ((product + nudge) / pow2_bits_minus_one)
                .try_into()
                .map_err(|_| "Error trying to convert LT::Double to LT")
                .unwrap();

            // assert(full_shift <= 31);

            // TODO: change inner_bit_and by & after the "InnerType split"
            let remainder = product_high & mask;
            let threshold = mask_div2 + is_negative;

            let core = (product_high >> effective_shift)
                + if remainder > threshold {
                    LT::ONE
                } else {
                    LT::ZERO
                };

            let shifted = core + output_zero_point;

            ST::try_from(partial_ord_clamp(
                shifted,
                LT::from(ST::MIN),
                LT::from(ST::MAX),
            ))
            .map_err(|_| "Unable to convert Large Type to Small Type")
            .unwrap()
        })
        .collect()
}

// Implementation of single-rounding requantisation with quantised parameters
pub fn requantize_single_round<ST, LT>(
    // TODO Think whether we can afford to pass ownership here and change the iter() below by into_iter()
    output: &[LT],
    effective_multiplier: LT,
    full_shift: usize,
    output_zero_point: ST,
) -> Vec<ST>
where
    ST: Integral + TryFrom<LT>,
    LT: Integral + From<ST>,
{
    // Although these parameters could be directly saved into the node (with
    // their final desired types), this computation/conversion is essentially
    // free. For the first two, storing them in the node with their actual types
    // (instead of the types needed for inference) makes the node more
    // transparent as far as definitions and proof system goes. TF Lite does the
    // same. For the other two, computing them here makes the function more
    // usable.
    let effective_multiplier = LT::Double::from(effective_multiplier);
    let output_zero_point = LT::from(output_zero_point);
    let non_neg_nudge = LT::ONE_DOUBLE << (full_shift - 1);
    let neg_nudge = non_neg_nudge - LT::ONE_DOUBLE;

    // Requantize
    // TODO add rayon for parallelization?
    output
        .iter()
        .map(|x| {
            let nudge = if *x >= LT::ZERO {
                non_neg_nudge
            } else {
                neg_nudge
            };

            let core: LT = ((LT::Double::from(*x) * effective_multiplier + nudge) >> full_shift)
                .try_into()
                .map_err(|_| "Error trying to convert LT::Double to LT")
                .unwrap();

            let shifted = core + output_zero_point;

            ST::try_from(partial_ord_clamp(
                shifted,
                LT::from(ST::MIN),
                LT::from(ST::MAX),
            ))
            .map_err(|_| "Unable to convert Large Type to Small Type")
            .unwrap()
        })
        .collect()
}

// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/quantization_util.cc#L53-L104
pub(crate) fn quantize_multiplier(double_multiplier: f64) -> (i32, usize) {
    if double_multiplier.is_zero() {
        return (0, 0);
    }

    let (q, expon) = frexp(double_multiplier);

    assert!(
        expon <= 0,
        "expon should be non-positive. Got: {} instead.",
        expon
    );

    // Negate expon to obtain the number of right-shift bits
    let mut shift = -expon as usize;

    // TF Lite uses C++'s round function under the hood as can be seen here:
    // https://github.com/tensorflow/tensorflow/blob/46f028f94dcd974705cd14e8abf05b9bd8f20bf0/tensorflow/lite/kernels/internal/cppmath.h#L35
    // The function rounds to the nearest integer, breaking ties away from zero.
    // The same strategy is implemented in Rust's round method:
    // https://doc.rust-lang.org/std/primitive.f64.html#method.round
    // See also: https://en.cppreference.com/w/c/numeric/fenv/FE_round
    let mut q_fixed = (q * ((1_i64 << (i32::BITS - 1)) as f64)).round() as i64;

    // TFLITE_CHECK(q_fixed <= (1LL << 31));
    assert!(
        q_fixed <= 1_i64 << (i32::BITS - 1),
        "q_fixed must not exceed 2^{}. Got: {} instead.",
        i32::BITS - 1,
        q_fixed
    );

    if q_fixed == 1_i64 << (i32::BITS - 1) {
        q_fixed /= 2;
        shift += 1;
    }

    // TFLITE_CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
    assert!(
        q_fixed <= i32::MAX as i64,
        "q_fixed must not exceed {}. Got: {} instead.",
        i32::MAX,
        q_fixed
    );

    // If exponent is too small.
    if expon < -((i32::BITS - 1) as isize) {
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
    assert!(x > 0.0);

    let x_bits: u64 = x.to_bits();

    // truncate low order bits to compute the biased exponent
    let mut expon = ((x_bits & F64_EXPONENT_MASK) / (1 << F64_EXPONENT_SHIFT)) as i32;

    // assert 0 < expon < 1023<<1
    assert!(expon > 0 && expon < ((F64_EXPONENT_BIAS as i32) << 1));

    // unbias exponent
    expon = expon - (F64_EXPONENT_BIAS as i32) + 1;

    let mantissa = x_bits & F64_FRACTION_MASK;

    let q = ((mantissa + (1 << F64_EXPONENT_SHIFT)) as f64)
        / ((1_u64 << (F64_EXPONENT_SHIFT + 1)) as f64);

    (q, expon as isize)
}

// This function is used to quantise model inputs and its types are fixed
pub fn quantise_f32_u8_nne(values: &[f32], scale: QScaleType, zero: u8) -> Vec<u8> {
    values
        .iter()
        .map(|x| {
            ((((*x as QScaleType) / scale) + (zero as f32)).round_ties_even() as i32)
                .clamp(u8::MIN as i32, u8::MAX as i32) as u8
        })
        .collect()
}
