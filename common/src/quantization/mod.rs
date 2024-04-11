use ark_std::Zero;

use crate::model::qarray::InnerType;

#[cfg(test)]
pub mod tests;

pub struct QInfo<ST, FT> {
    pub scale: FT,
    pub zero_point: ST,
}

// TODO: this will probably change to inference-ready requantisation info
// Even what is being done now could be optimised by precomputing outside the
// evaluate function
pub struct BMMQInfo<ST, FT> {
    pub input_info: QInfo<ST, FT>,
    pub weight_info: QInfo<ST, FT>,
    // Bias requantisation information is not used (and is indeed directly
    // computable from the two above)
    pub output_info: QInfo<ST, FT>,
}

pub enum RoundingScheme {
    NearestTiesAwayFromZero,
    NearestTiesEven,
}

pub trait RoundInt<OT> {
    fn round(self) -> OT;
    fn round_ties_even(self) -> OT;
}

impl RoundInt<i32> for f64 {
    fn round(self) -> i32 {
        self.round() as i32
    }

    fn round_ties_even(self) -> i32 {
        self.round_ties_even() as i32
    }
}

pub fn requantise_fc<ST, LT, FT>(
    output: &[LT],
    q_info: &BMMQInfo<ST, FT>,
    scheme: RoundingScheme,
) -> Vec<ST>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
    FT: InnerType + From<LT>,
    FT::Double: From<LT> + RoundInt<LT>,
{
    match scheme {
        RoundingScheme::NearestTiesAwayFromZero => {
            requantise_fc_ntafz::<ST, LT, FT>(output, q_info)
        }
        RoundingScheme::NearestTiesEven => requantise_fc_nte::<ST, LT, FT>(output, q_info),
    }
}

fn requantise_fc_ntafz<ST, LT, FT>(output: &[LT], q_info: &BMMQInfo<ST, FT>) -> Vec<ST>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
    FT: InnerType + From<LT>,
    FT::Double: From<LT> + RoundInt<LT>,
{
    // 1. Computing scale
    // TODO In actual schemes, this will be decomposed as (int, shift)
    let (s_i, s_w, s_o) = (
        q_info.input_info.scale,
        q_info.weight_info.scale,
        q_info.output_info.scale,
    );
    let (s_i, s_w, s_o) = (
        <FT::Double as From<FT>>::from(s_i),
        <FT::Double as From<FT>>::from(s_w),
        <FT::Double as From<FT>>::from(s_o),
    );
    let s = s_i * s_w / s_o;

    // 2. Requantise
    // TODO add rayon for parallelisation?
    output
        .iter()
        .map(|x| {
            let x = (FT::Double::from(*x) * s).round();
            x += LT::from(q_info.output_info.zero_point);
            ST::try_from(partial_ord_clamp(x, LT::from(ST::MIN), LT::from(ST::MAX)))
                .map_err(|_| "Unable to convert Large Type to Small Type")
                .unwrap()
        })
        .collect()
}

// The (unstable) method clamp comes from the trait Ord, which we cannot
// restrict InnerType to as we need f32 to implement the latter. Note that this
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

fn requantise_fc_nte<ST, LT, FT>(output: &[LT], q_info: &BMMQInfo<ST, FT>) -> Vec<ST>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
    FT: InnerType + From<LT>,
    FT::Double: From<LT> + RoundInt<LT>,
{
    // 1. Computing scale
    // TODO In actual schemes, this will be decomposed as (int, shift)
    let (s_i, s_w, s_o) = (
        q_info.input_info.scale,
        q_info.weight_info.scale,
        q_info.output_info.scale,
    );
    let (s_i, s_w, s_o) = (
        <FT::Double as From<FT>>::from(s_i),
        <FT::Double as From<FT>>::from(s_w),
        <FT::Double as From<FT>>::from(s_o),
    );
    let s = s_i * s_w / s_o;

    // 2. Requantise
    // TODO add rayon for parallelisation?
    output
        .iter()
        .map(|x| {
            let x = (FT::Double::from(*x) * s).round_ties_even();
            x += LT::from(q_info.output_info.zero_point);
            ST::try_from(partial_ord_clamp(x, LT::from(ST::MIN), LT::from(ST::MAX)))
                .map_err(|_| "Unable to convert Large Type to Small Type")
                .unwrap()
        })
        .collect()
}

// Implementation of TF Lite's reference requantization.
pub fn requantise_ref<ST, LT>(
    // TODO Think whether we can afford to pass ownership here and change the iter() below by into_iter()
    output: &[LT],
    effective_multiplier: LT,
    effective_shift: usize,
    output_zero_point: ST,
) -> Vec<ST>
where
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
{
    // Computing auxiliary constants used for every input
    let effective_multiplier = LT::Double::from(effective_multiplier);
    let output_zero_point = LT::from(output_zero_point);
    let pow2_effective_shift = LT::pow2(effective_shift);
    let xt_pow2_bits_minus_one = LT::Double::from(LT::pow2(LT::BITS - 1));

    // Mask consists of effective_shift ones
    let mask = LT::pow2(effective_shift) - LT::ONE;
    let mask_div2 = mask / LT::TWO;

    // Constants used during nudging
    let non_neg_nudge = LT::pow2(LT::BITS - 2);
    let neg_nudge = LT::ONE - LT::pow2(LT::BITS - 2);

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

            let x = LT::Double::from(*x) * effective_multiplier;

            let x_high =
                LT::inner_try_from((LT::Double::from(nudge) + x) / xt_pow2_bits_minus_one).unwrap();

            // assert(right_shift <= 31);

            // TODO: change inner_bit_and by & after the "InnerType split"
            let remainder = x_high.inner_bit_and(mask);
            let threshold = mask_div2 + is_negative;

            let out = x_high / pow2_effective_shift
                + if remainder > threshold {
                    LT::ONE
                } else {
                    LT::ZERO
                };

            let shifted_out = out + output_zero_point;

            ST::try_from(partial_ord_clamp(
                shifted_out,
                LT::from(ST::MIN),
                LT::from(ST::MAX),
            ))
            .map_err(|_| "Unable to convert Large Type to Small Type")
            .unwrap()
        })
        .collect()
}

//
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
    if q_fixed > 1_i64 << (i32::BITS - 1) {
        panic!(
            "q_fixed must not exceed {}. Got: {} instead.",
            i32::BITS - 1,
            q_fixed
        );
    }

    if q_fixed == (1_i64 << (i32::BITS - 1)) {
        // 1 << 31
        q_fixed /= 2;
        shift += 1;
    }

    // TFLITE_CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
    if q_fixed > i32::MAX as i64 {
        panic!(
            "q_fixed must not exceed {}. Got: {} instead.",
            i32::MAX,
            q_fixed
        );
    }

    // If exponent is too small.
    // if (-expon as u32) < i32::BITS - 1 {
    if expon < -((i32::BITS - 1) as isize) {
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
    // expon -= (F64_EXPONENT_BIAS as i32) + 1;
    expon = expon - (F64_EXPONENT_BIAS as i32) + 1;

    let mantissa = x_bits & F64_FRACTION_MASK;

    let q = ((mantissa + (1 << F64_EXPONENT_SHIFT)) as f64)
        / ((1_u64 << (F64_EXPONENT_SHIFT + 1)) as f64);

    (q, expon as isize)
}

// This function is used to quantise model model inputs and its types are fixed
pub fn quantise_f32_u8_nne(values: &[f32], scale: f32, zero: u8) -> Vec<u8> {
    values
        .iter()
        .map(|x| {
            ((((*x as f32) / scale) + (zero as f32)).round_ties_even() as i32)
                .clamp(u8::MIN as i32, u8::MAX as i32) as u8
        })
        .collect()
}
